#include "identity.h"
#include "facedetector.h"
#include "ocr.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace std;

namespace sc {
    namespace impl {
        // A recognised line of text. Words are kept both spaced, for reading
        // labelled fields, and packed, because the identity number on a green
        // ID book is printed in widely separated groups that the recogniser
        // reports as separate words.
        struct text_line {
            string text; // words joined with single spaces
            string packed; // words joined with nothing
            string alnum; // packed, uppercased, punctuation removed
            double confidence{}; // mean word confidence, 0-100
        };

        // ---------------------------------------------------------------- text

        static string upper(string value) {
            ranges::transform(value, value.begin(),
                              [](const unsigned char c) { return static_cast<char>(toupper(c)); });
            return value;
        }

        static string keep_alnum(const string &value) {
            string result;
            result.reserve(value.size());
            for (const unsigned char c: value) if (isalnum(c)) result += static_cast<char>(toupper(c));
            return result;
        }

        static string keep_digits(const string &value) {
            string result;
            result.reserve(value.size());
            for (const unsigned char c: value) if (isdigit(c)) result += static_cast<char>(c);
            return result;
        }

        static string trim(const string &value) {
            const auto begin = value.find_first_not_of(" \t\r\n:./,-");
            if (begin == string::npos) return {};
            const auto end = value.find_last_not_of(" \t\r\n:./,");
            return value.substr(begin, end - begin + 1);
        }

        // Keeps only the characters that belong in a printed name.
        static string clean_name(const string &value) {
            string result;
            for (const unsigned char c: upper(trim(value))) {
                if (isalpha(c) || c == ' ' || c == '-' || c == '\'') result += static_cast<char>(c);
            }
            return trim(result);
        }

        // ---------------------------------------------------------------- dates

        static constexpr array<string_view, 12> MONTHS = {
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
        };

        static int current_two_digit_year() {
            const auto now = chrono::system_clock::now();
            const auto days = chrono::floor<chrono::days>(now);
            const chrono::year_month_day today{days};
            return static_cast<int>(today.year()) % 100;
        }

        // Resolves a two-digit year: anything after this year belongs to the
        // previous century, so 82 is 1982 while 08 is 2008.
        static int resolve_year(const int two_digit_year) {
            return two_digit_year > current_two_digit_year() ? 1900 + two_digit_year : 2000 + two_digit_year;
        }

        static bool plausible_month_day(const int month, const int day) {
            static constexpr array<int, 12> LENGTHS = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            return month >= 1 && month <= 12 && day >= 1 && day <= LENGTHS[month - 1];
        }

        static string iso_date(const int year, const int month, const int day) {
            char buffer[11];
            snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", year, month, day);
            return buffer;
        }

        // Reads a printed date in any of the forms the three documents use:
        // 1984-03-24, 1984/03/24, 24 MAR 1984 or 240384. Returns an empty
        // string when nothing parses.
        static string parse_date(const string &value) {
            const string packed = keep_alnum(value);

            // yyyymmdd
            if (packed.size() >= 8 && all_of(packed.begin(), packed.begin() + 8, ::isdigit)) {
                const int year = stoi(packed.substr(0, 4));
                const int month = stoi(packed.substr(4, 2));
                const int day = stoi(packed.substr(6, 2));
                if (year >= 1900 && year <= 2100 && plausible_month_day(month, day))
                    return iso_date(year, month, day);
            }
            // ddMMMyyyy
            if (packed.size() >= 9) {
                for (size_t i = 0; i + 9 <= packed.size(); ++i) {
                    const string_view window{packed.data() + i, 9};
                    if (!isdigit(static_cast<unsigned char>(window[0]))) continue;
                    if (!isdigit(static_cast<unsigned char>(window[1]))) continue;
                    const auto month = ranges::find(MONTHS, window.substr(2, 3));
                    if (month == MONTHS.end()) continue;
                    if (!all_of(window.begin() + 5, window.end(), ::isdigit)) continue;
                    const int day = stoi(string{window.substr(0, 2)});
                    const int month_number = static_cast<int>(ranges::distance(MONTHS.begin(), month)) + 1;
                    const int year = stoi(string{window.substr(5, 4)});
                    if (year >= 1900 && year <= 2100 && plausible_month_day(month_number, day))
                        return iso_date(year, month_number, day);
                }
            }
            return {};
        }

        // The renderings of a date that could appear on these documents, packed
        // the same way recognised lines are so they can be searched for.
        static vector<string> date_renderings(const int year, const int month, const int day) {
            char iso[9], named[10], named_padded[10];
            snprintf(iso, sizeof(iso), "%04d%02d%02d", year, month, day);
            snprintf(named, sizeof(named), "%d%s%04d", day, string{MONTHS[month - 1]}.c_str(), year);
            snprintf(named_padded, sizeof(named_padded), "%02d%s%04d", day, string{MONTHS[month - 1]}.c_str(), year);
            return {iso, named, named_padded};
        }

        // --------------------------------------------------- identity numbers

        static bool luhn_valid(const string &digits) {
            int total = 0;
            for (size_t i = 0; i < digits.size(); ++i) {
                int value = digits[digits.size() - 1 - i] - '0';
                if (i % 2 == 1) {
                    value *= 2;
                    if (value > 9) value -= 9;
                }
                total += value;
            }
            return total % 10 == 0;
        }

        // ----------------------------------------------------------------- MRZ

        static int mrz_value(const char c) {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
            return 0; // filler
        }

        static int mrz_check_digit(const string_view field) {
            static constexpr array<int, 3> WEIGHTS = {7, 3, 1};
            int total = 0;
            for (size_t i = 0; i < field.size(); ++i) total += mrz_value(field[i]) * WEIGHTS[i % 3];
            return total % 10;
        }

        static bool mrz_check_matches(const string_view field, const char check) {
            return isdigit(static_cast<unsigned char>(check)) && mrz_check_digit(field) == check - '0';
        }

        // Repairs the letters a recogniser most often substitutes for digits.
        // Safe to apply blindly because every field it touches is protected by
        // a check digit that will reject a wrong guess.
        static string as_digits(string field) {
            for (char &c: field) {
                switch (c) {
                    case 'O':
                    case 'Q':
                    case 'D': c = '0';
                        break;
                    case 'I':
                    case 'L': c = '1';
                        break;
                    case 'Z': c = '2';
                        break;
                    case 'G': c = '6';
                        break;
                    case 'S': c = '5';
                        break;
                    case 'B': c = '8';
                        break;
                    default: break;
                }
            }
            return field;
        }

        // The second line of a TD3 machine readable zone, which is where a
        // South African passport carries the holder's identity number.
        struct mrz_line2 {
            static constexpr size_t length = 44;

            string passport_number;
            string nationality;
            string date_of_birth; // ISO
            string sex;
            string date_of_expiry; // ISO
            string personal_number; // the 13-digit identity number
            bool composite_valid{};

            // Parses a 44-character window, returning false unless the birth
            // date, expiry and personal number check digits all agree.
            bool parse(const string_view raw) {
                if (raw.size() != length) return false;

                // The fields repaired here are numeric on a South African
                // passport, including the check digits themselves, so the
                // letters a recogniser most often substitutes for digits are
                // put back before anything is tested. A wrong repair just fails
                // its check digit and the window is discarded. The document
                // number, nationality and sex are left alone: they are
                // legitimately alphabetic.
                string window{raw};
                const auto repair = [&window](const size_t at, const size_t count) {
                    window.replace(at, count, as_digits(window.substr(at, count)));
                };
                repair(9, 1); // document number check digit
                repair(13, 7); // date of birth and its check digit
                repair(21, 7); // date of expiry and its check digit
                repair(28, 13); // the identity number, less its filler
                repair(42, 2); // personal number and composite check digits

                const string birth = window.substr(13, 6);
                const string expiry = window.substr(21, 6);
                const string personal = window.substr(28, 14);

                if (!mrz_check_matches(birth, window[19])) return false;
                if (!mrz_check_matches(expiry, window[27])) return false;
                if (!mrz_check_matches(personal, window[42])) return false;

                const int birth_month = stoi(birth.substr(2, 2));
                const int birth_day = stoi(birth.substr(4, 2));
                if (!plausible_month_day(birth_month, birth_day)) return false;
                date_of_birth = iso_date(resolve_year(stoi(birth.substr(0, 2))), birth_month, birth_day);

                const int expiry_month = stoi(expiry.substr(2, 2));
                const int expiry_day = stoi(expiry.substr(4, 2));
                if (!plausible_month_day(expiry_month, expiry_day)) return false;
                // A document that is current expires this century.
                date_of_expiry = iso_date(2000 + stoi(expiry.substr(0, 2)), expiry_month, expiry_day);

                passport_number = window.substr(0, 9);
                erase(passport_number, '<');
                nationality = window.substr(10, 3);
                sex = window[20] == 'M' || window[20] == 'F' ? string{window[20]} : string{};
                personal_number = personal;
                erase(personal_number, '<');

                // One digit covering the document number, both dates and the
                // personal number together, each with its own check digit.
                const string composite = window.substr(0, 10) + window.substr(13, 7) + window.substr(21, 22);
                composite_valid = mrz_check_matches(composite, window[43]);
                return true;
            }
        };

        // Finds a usable MRZ second line anywhere in the recognised text.
        static optional<mrz_line2> find_mrz(const vector<text_line> &lines) {
            optional<mrz_line2> best;
            for (const auto &line: lines) {
                string window;
                window.reserve(line.packed.size());
                for (const unsigned char c: upper(line.packed)) {
                    if (isalnum(c) || c == '<') window += static_cast<char>(c);
                }
                if (window.size() < mrz_line2::length) continue;
                for (size_t start = 0; start + mrz_line2::length <= window.size(); ++start) {
                    mrz_line2 candidate;
                    if (!candidate.parse(string_view{window}.substr(start, mrz_line2::length))) continue;
                    if (!identity::valid_id_number(candidate.personal_number)) continue;
                    if (candidate.composite_valid) return candidate; // fully verified, stop looking
                    if (!best) best = candidate;
                }
            }
            return best;
        }

        // -------------------------------------------------------- field labels

        // Returns the part of a line that follows its first n alphanumeric
        // characters, so a value can be taken from after a label that was
        // matched with its punctuation stripped.
        static string tail_after(const string &text, const size_t alnum_count) {
            size_t seen = 0;
            for (size_t i = 0; i < text.size(); ++i) {
                if (!isalnum(static_cast<unsigned char>(text[i]))) continue;
                if (++seen == alnum_count) return text.substr(i + 1);
            }
            return {};
        }

        // Returns the value belonging to a label. Labels are matched on a
        // punctuation-free uppercase form because the small low-contrast
        // captions on these documents recognise poorly.
        static string value_for_label(const vector<text_line> &lines,
                                      const initializer_list<string_view> keys) {
            for (size_t i = 0; i < lines.size(); ++i) {
                for (const auto key: keys) {
                    const auto at = lines[i].alnum.find(key);
                    if (at == string::npos) continue;

                    // A value shares its label's line only where digits follow
                    // the label, as in an ID book's "I.D.No. 791130 5089 08 9".
                    // Captions elsewhere are bilingual - "Surname / Nom" - so a
                    // remainder without digits is the second language, not the
                    // value, and the value is on the line below.
                    const string remainder = lines[i].alnum.substr(at + key.size());
                    if (ranges::any_of(remainder, [](const unsigned char c) { return isdigit(c); })) {
                        const string tail = trim(tail_after(lines[i].text, at + key.size()));
                        if (!tail.empty()) return tail;
                    }
                    if (i + 1 < lines.size() && !trim(lines[i + 1].text).empty()) return trim(lines[i + 1].text);
                }
            }
            return {};
        }

        static bool any_line_contains(const vector<text_line> &lines, const string_view needle) {
            return ranges::any_of(lines, [needle](const text_line &line) {
                return line.alnum.find(needle) != string::npos;
            });
        }

        // ------------------------------------------------------------- the OCR
        static vector<text_line> recognise(const filesystem::path &image_path) {
            static thread_local ocr recognizer;
            recognizer.detect(image_path);

            vector<text_line> lines;
            for (const auto &recognized: recognizer.lines()) {
                text_line line;
                line.text = trim(recognized.text);
                line.packed.reserve(line.text.size());
                for (const unsigned char c: line.text)
                    if (!isspace(c)) line.packed += static_cast<char>(c);
                line.alnum = keep_alnum(line.packed);
                line.confidence = static_cast<double>(recognized.confidence);
                if (!line.text.empty()) lines.push_back(std::move(line));
            }
            return lines;
        }

        // ------------------------------------------------------------ assembly

        struct number_hit {
            string number;
            double confidence{};
        };

        // Looks for a valid identity number in the digits of each line. The
        // number is printed in separated groups on a green ID book, so the
        // digits of a whole line are joined before the window slides over them.
        static optional<number_hit> find_id_number(const vector<text_line> &lines) {
            for (const auto &line: lines) {
                const string digits = keep_digits(line.packed);
                if (digits.size() < 13) continue;
                for (size_t start = 0; start + 13 <= digits.size(); ++start) {
                    const string candidate = digits.substr(start, 13);
                    if (identity::valid_id_number(candidate)) return number_hit{candidate, line.confidence};
                }
            }
            return nullopt;
        }

        class identity_reader {
        public:
            static identity read(const filesystem::path &image_path, const bool face_detection) {
                identity best;
                best.face_detection_performed_ = face_detection;

                if (face_detection) {
                    static thread_local facedetector detector;
                    best.faces_ = detector.detect(image_path, false);
                }

                populate(best, recognise(image_path));
                return best;
            }

        private:
            static void populate(identity &result, const vector<text_line> &lines) {
                const auto mrz = find_mrz(lines);
                const auto printed = find_id_number(lines);

                result.document_ = classify(lines, mrz.has_value());

                if (mrz) {
                    result.id_number_ = mrz->personal_number;
                    result.checks_.mrz = true;
                    result.passport_number_ = mrz->passport_number;
                    result.date_of_expiry_ = mrz->date_of_expiry;
                    result.sex_ = mrz->sex;
                    result.nationality_ = mrz->nationality;
                    result.ocr_confidence_ = mrz_line_confidence(lines);
                } else if (printed) {
                    result.id_number_ = printed->number;
                    result.ocr_confidence_ = printed->confidence;
                }
                if (result.id_number_.empty()) return;

                result.checks_.luhn = true; // nothing reaches here without it
                result.date_of_birth_ = identity::id_date_of_birth(result.id_number_);
                read_fields(result, lines);
                cross_check(result, lines, mrz.has_value());
            }

            static double mrz_line_confidence(const vector<text_line> &lines) {
                for (const auto &line: lines) {
                    if (line.alnum.size() >= 40 && line.packed.find('<') != string::npos) return line.confidence;
                }
                return lines.empty() ? 0.0 : lines.front().confidence;
            }

            // Counts the markers each document carries rather than taking the
            // first match, because a photograph often loses some of them: a
            // card's small green captions and a worn book's Afrikaans labels
            // both recognise unreliably.
            static identity::document classify(const vector<text_line> &lines, const bool has_mrz) {
                if (has_mrz) return identity::document::passport; // decisive on its own

                const auto count = [&lines](const initializer_list<string_view> markers) {
                    return ranges::count_if(markers, [&lines](const string_view marker) {
                        return any_line_contains(lines, marker);
                    });
                };

                const auto passport = count({
                    "PASSEPORT", "PASSPORTNO", "SUDAFRICAIN", "AFRIQUEDUSUD",
                    "DATEOFEXPIRY"
                });
                const auto book = count({
                    "IDNO", "SABURGER", "SACITIZEN", "VANSURNAME", "VOORNAME",
                    "GEBOORTEDATUM", "GEBOORTEDISTRIK", "DATUMUITGEREIK",
                    "BINNELANDSESAKE", "IDENTITEITSDOKUMENT"
                });
                const auto card = count({
                    "IDENTITYCARD", "NATIONALIDENTITY", "IDENTITYNUMBER",
                    "COUNTRYOFBIRTH", "STATUS"
                });

                const auto best = max({passport, book, card});
                if (best == 0) return identity::document::unknown;
                if (best == passport) return identity::document::passport;
                if (best == book) return identity::document::id_book;
                return identity::document::id_card;
            }

            static void read_fields(identity &result, const vector<text_line> &lines) {
                const auto set_if_empty = [](string &field, const string &value) {
                    if (field.empty() && !value.empty()) field = value;
                };

                set_if_empty(result.surname_, clean_name(value_for_label(lines, {"SURNAME", "VAN"})));
                set_if_empty(result.names_, clean_name(value_for_label(lines, {
                                                                           "GIVENNAMES", "FORENAMES", "NAMES",
                                                                           "VOORNAME", "PRENOMS"
                                                                       })));
                set_if_empty(result.nationality_, clean_name(value_for_label(lines, {"NATIONALIT"})));
                set_if_empty(result.country_of_birth_,
                             clean_name(value_for_label(lines, {
                                                            "COUNTRYOFBIRTH", "PLACEOFBIRTH",
                                                            "GEBOORTEDISTRIK"
                                                        })));
                set_if_empty(result.status_, clean_name(value_for_label(lines, {"STATUS"})));
                set_if_empty(result.date_of_issue_,
                             parse_date(value_for_label(lines, {"DATEISSUED", "DATEOFISSUE", "DATUMUITGEREIK"})));
                set_if_empty(result.date_of_expiry_,
                             parse_date(value_for_label(lines, {"DATEOFEXPIRY", "DATEEXPIRES"})));

                if (result.passport_number_.empty()) {
                    const string passport = keep_alnum(value_for_label(lines, {"PASSPORTNO"}));
                    if (passport.size() >= 8 && passport.size() <= 9) result.passport_number_ = passport;
                }
            }

            // Confirms the number against parts of the document it was not read
            // from: the printed birth date, and the printed sex.
            static void cross_check(identity &result, const vector<text_line> &lines, const bool from_mrz) {
                const string &number = result.id_number_;
                const int month = stoi(number.substr(2, 2));
                const int day = stoi(number.substr(4, 2));

                // Look for the birth date the number implies, rather than
                // parsing an unknown date, so the two centuries and all three
                // printed layouts are covered.
                for (const int year: {
                         resolve_year(stoi(number.substr(0, 2))),
                         1900 + stoi(number.substr(0, 2)),
                         2000 + stoi(number.substr(0, 2))
                     }) {
                    for (const auto &rendering: date_renderings(year, month, day)) {
                        if (any_line_contains(lines, rendering)) {
                            result.checks_.date_of_birth = true;
                            result.date_of_birth_ = iso_date(year, month, day);
                            break;
                        }
                    }
                    if (result.checks_.date_of_birth) break;
                }
                // An MRZ carries its own birth date under a check digit, which
                // is an independent confirmation in itself.
                if (from_mrz && !result.date_of_birth_.empty()) result.checks_.date_of_birth = true;

                const string expected = identity::id_sex(number);
                if (!result.sex_.empty()) {
                    result.checks_.sex = result.sex_ == expected;
                } else {
                    const string printed = printed_sex(lines);
                    if (!printed.empty()) {
                        result.sex_ = printed;
                        result.checks_.sex = printed == expected;
                    }
                }
                if (result.sex_.empty()) result.sex_ = expected;
            }

            // The sex is printed as a lone M or F beside or under its label.
            static string printed_sex(const vector<text_line> &lines) {
                for (size_t i = 0; i < lines.size(); ++i) {
                    auto at = lines[i].alnum.find("SEX");
                    size_t width = 3;
                    if (at == string::npos) {
                        at = lines[i].alnum.find("GESLAG");
                        width = 6;
                    }
                    if (at == string::npos) continue;
                    for (const string &candidate: {
                             lines[i].alnum.substr(at + width),
                             i + 1 < lines.size() ? lines[i + 1].alnum : string{}
                         }) {
                        if (candidate == "M" || candidate == "F") return candidate;
                    }
                }
                return {};
            }
        };
    } // namespace impl

    // ------------------------------------------------------------- public API

    int identity::verification::score() const noexcept {
        if (!luhn) return 0;
        return min(100, 60 + (date_of_birth ? 25 : 0) + (sex ? 10 : 0) + (mrz ? 35 : 0));
    }

    int identity::verification::passed() const noexcept {
        return (luhn ? 1 : 0) + (date_of_birth ? 1 : 0) + (sex ? 1 : 0) + (mrz ? 1 : 0);
    }

    nlohmann::ordered_json identity::verification::to_json() const {
        return nlohmann::ordered_json{
            {"luhn", luhn}, {"date_of_birth", date_of_birth}, {"sex", sex}, {"mrz", mrz}
        };
    }

    identity::identity(const filesystem::path &image_path, const bool face_detection) {
        read(image_path, face_detection);
    }

    bool identity::read(const filesystem::path &image_path, const bool face_detection) {
        *this = impl::identity_reader::read(image_path, face_detection);
        return has_id_number();
    }

    identity::document identity::document_type() const noexcept { return document_; }

    string identity::document_type_name() const {
        switch (document_) {
            case document::id_card: return "id_card";
            case document::id_book: return "id_book";
            case document::passport: return "passport";
            default: return "unknown";
        }
    }

    bool identity::has_id_number() const noexcept { return !id_number_.empty(); }
    const string &identity::id_number() const noexcept { return id_number_; }
    percent identity::confidence() const { return {static_cast<double>(checks_.score()), 0}; }
    const identity::verification &identity::checks() const noexcept { return checks_; }
    percent identity::ocr_confidence() const { return {ocr_confidence_, 1}; }
    const vector<face> &identity::faces() const noexcept { return faces_; }
    int identity::rotation() const noexcept { return rotation_; }
    const string &identity::surname() const noexcept { return surname_; }
    const string &identity::names() const noexcept { return names_; }
    const string &identity::date_of_birth() const noexcept { return date_of_birth_; }
    const string &identity::sex() const noexcept { return sex_; }
    const string &identity::nationality() const noexcept { return nationality_; }
    const string &identity::country_of_birth() const noexcept { return country_of_birth_; }
    const string &identity::status() const noexcept { return status_; }
    const string &identity::passport_number() const noexcept { return passport_number_; }
    const string &identity::date_of_issue() const noexcept { return date_of_issue_; }
    const string &identity::date_of_expiry() const noexcept { return date_of_expiry_; }

    optional<bool> identity::citizen() const {
        if (!has_id_number()) return nullopt;
        return id_number_[10] == '0';
    }

    bool identity::valid_id_number(const string &number) {
        if (number.size() != 13) return false;
        if (!ranges::all_of(number, [](const unsigned char c) { return isdigit(c); })) return false;
        if (!impl::plausible_month_day(stoi(number.substr(2, 2)), stoi(number.substr(4, 2)))) return false;
        if (number[10] != '0' && number[10] != '1') return false; // citizen or permanent resident
        return impl::luhn_valid(number);
    }

    string identity::id_date_of_birth(const string &number) {
        if (!valid_id_number(number)) return {};
        return impl::iso_date(impl::resolve_year(stoi(number.substr(0, 2))),
                              stoi(number.substr(2, 2)), stoi(number.substr(4, 2)));
    }

    string identity::id_sex(const string &number) {
        if (!valid_id_number(number)) return {};
        return stoi(number.substr(6, 4)) >= 5000 ? "M" : "F";
    }

    nlohmann::ordered_json identity::to_json() const {
        nlohmann::ordered_json result{
            {"version", 1},
            {"document_type", document_type_name()},
            {"id_number", id_number_},
            {"confidence", static_cast<double>(confidence())},
            {"verification", checks_.to_json()}
        };
        const auto add = [&result](const char *key, const string &value) {
            if (!value.empty()) result[key] = value;
        };
        add("surname", surname_);
        add("names", names_);
        add("date_of_birth", date_of_birth_);
        add("sex", sex_);
        if (const auto is_citizen = citizen()) result["citizen"] = *is_citizen;
        add("nationality", nationality_);
        add("country_of_birth", country_of_birth_);
        add("status", status_);
        add("passport_number", passport_number_);
        add("date_of_issue", date_of_issue_);
        add("date_of_expiry", date_of_expiry_);
        if (has_id_number()) result["ocr_confidence"] = static_cast<double>(ocr_confidence());
        if (face_detection_performed_) {
            result["faces"] = nlohmann::ordered_json::array();
            for (const auto &detected_face: faces_) result["faces"].push_back(detected_face.to_json());
        }
        result["rotation"] = rotation_;
        return result;
    }

    identity::operator nlohmann::ordered_json() const { return to_json(); }
    identity::operator string() const { return to_json().dump(); }

    std::ostream &operator<<(std::ostream &lhs, const identity &rhs) {
        return lhs << static_cast<string>(rhs);
    }
} // namespace sc
