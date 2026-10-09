#include "identity.h"
#include <svg2png.h>
#include <sc_test.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

namespace {
    constexpr auto VALID = "8001015009087";
    constexpr auto VALID_FEMALE = "9912310001083";
    constexpr auto BAD_CHECK_DIGIT = "8001015009086";
    constexpr auto BAD_MONTH = "8013015009082";
    constexpr auto BAD_CITIZENSHIP = "8001015009285";

    // An MRZ check digit: weights 7, 3, 1; digits as themselves, letters from 10, fillers 0.
    char check_digit(const string &field) {
        int sum = 0;
        for (size_t i = 0; i < field.size(); ++i) {
            const char c = field[i];
            const int value = isdigit(c) ? c - '0' : isalpha(c) ? c - 'A' + 10 : 0;
            sum += value * (i % 3 == 0 ? 7 : i % 3 == 1 ? 3 : 1);
        }
        return static_cast<char>('0' + sum % 10);
    }
}

int main() {
    SECTION("Identity number validation");
    {
        CHECK(sc::identity::valid_id_number(VALID));
        CHECK(sc::identity::valid_id_number(VALID_FEMALE));
        CHECK(!sc::identity::valid_id_number(BAD_CHECK_DIGIT));
        CHECK(!sc::identity::valid_id_number(BAD_MONTH));
        CHECK(!sc::identity::valid_id_number(BAD_CITIZENSHIP));
        CHECK(!sc::identity::valid_id_number(""));
        CHECK(!sc::identity::valid_id_number("800101500908"));
        CHECK(!sc::identity::valid_id_number("80010150090855"));
        CHECK(!sc::identity::valid_id_number("80010150090X5"));
    }

    SECTION("Fields derived from an identity number");
    {
        CHECK_EQ(sc::identity::id_date_of_birth(VALID), string{"1980-01-01"});
        CHECK_EQ(sc::identity::id_date_of_birth(VALID_FEMALE), string{"1999-12-31"});
        CHECK_EQ(sc::identity::id_sex(VALID), string{"M"});
        CHECK_EQ(sc::identity::id_sex(VALID_FEMALE), string{"F"});
        CHECK_EQ(sc::identity::id_date_of_birth(BAD_CHECK_DIGIT), string{});
        CHECK_EQ(sc::identity::id_sex(BAD_CHECK_DIGIT), string{});
    }

    SECTION("Empty result");
    {
        const sc::identity nothing;
        CHECK(!nothing.has_id_number());
        CHECK(nothing.document_type() == sc::identity::document::unknown);
        CHECK_EQ(nothing.document_type_name(), string{"unknown"});
        CHECK_EQ(static_cast<double>(nothing.confidence()), 0.0);
        CHECK(!nothing.citizen().has_value());
        CHECK_EQ(nothing.to_json()["id_number"].get<string>(), string{});
    }

    SECTION("Reading a passport");
    {
        // A specimen with made-up data (resource/samples).
        const sc::identity passport{"resource/samples/passport.jpg"};
        CHECK_EQ(passport.id_number(), string{VALID});
        CHECK_EQ(passport.surname(), string{"SAMPLESON"});
        CHECK_EQ(passport.names(), string{"ALEX JORDAN"});
        CHECK_EQ(passport.passport_number(), string{"A12345678"});
        CHECK(passport.has_id_number());
        CHECK(passport.document_type() == sc::identity::document::passport);
        CHECK(sc::identity::valid_id_number(passport.id_number()));
        CHECK(passport.checks().mrz);
        CHECK(passport.checks().luhn);
        CHECK_BETWEEN(static_cast<double>(passport.confidence()), 90.0, 100.0);
        CHECK(!passport.passport_number().empty());
        CHECK(!passport.date_of_expiry().empty());
        CHECK_EQ(passport.date_of_birth(), sc::identity::id_date_of_birth(passport.id_number()));
        CHECK_EQ(passport.sex(), sc::identity::id_sex(passport.id_number()));
    }

    SECTION("Reading a passport's machine readable zone, names with O in them");
    {
        // Made up. In a monospace font O and 0 look alike: a name field takes only letters.
        const string number = "A12345678", birth = "800101", expiry = "320101", personal = string{VALID} + "<";
        string line2 = number + check_digit(number) + "ZAF" + birth + check_digit(birth) + "M" + expiry +
                       check_digit(expiry) + personal + check_digit(personal);
        line2 += check_digit(line2.substr(0, 10) + line2.substr(13, 7) + line2.substr(21, 22));
        string line1 = "P<ZAFSAMPLESON<<ALEX<JORDAN";
        line1.append(44 - line1.size(), '<');
        const auto escaped = [](string text) {
            string out;
            for (const char c: text) out += c == '<' ? string{"&lt;"} : string{c};
            return out;
        };
        const string svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"260\">"
                           "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
                           "<g font-family=\"OCR-B, DejaVu Sans Mono, Menlo, Courier New, monospace\" font-size=\"36\">"
                           "<text x=\"40\" y=\"110\">" + escaped(line1) + "</text>"
                           "<text x=\"40\" y=\"180\">" + escaped(line2) + "</text></g></svg>";
        const auto png = sc::svg2png::FromString(svg);
        if (png.size() < 3000) {
            cout << "   (no font to render text with: skipped)" << endl;
        } else {
            const auto path = (filesystem::temp_directory_path() / "sc-ai-test-mrz.png").string();
            ofstream{path, ios::binary} << png;
            const sc::identity passport{filesystem::path{path}};
            filesystem::remove(path);
            CHECK(passport.document_type() == sc::identity::document::passport);
            CHECK_EQ(passport.id_number(), string{VALID});
            CHECK_EQ(passport.surname(), string{"SAMPLESON"});
            CHECK_EQ(passport.names(), string{"ALEX JORDAN"});
            CHECK_EQ(passport.passport_number(), number);
        }
    }

    SECTION("A passport's heading is never read as a name");
    {
        // Made up. Without an MRZ the names come from the labels; the bilingual heading, a word
        // the length of a name in capitals, was taken for them.
        const string svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"520\">"
                           "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
                           "<g font-family=\"Arial, Helvetica, DejaVu Sans, Liberation Sans, sans-serif\">"
                           "<text x=\"40\" y=\"70\" font-size=\"34\">PASSPORT PASPOORT PASSEPORT PASAPORTE</text>"
                           "<text x=\"40\" y=\"150\" font-size=\"18\">Surname / Van</text>"
                           "<text x=\"40\" y=\"185\" font-size=\"28\">SAMPLESON</text>"
                           "<text x=\"40\" y=\"250\" font-size=\"18\">Given names / Voorname</text>"
                           "<text x=\"40\" y=\"285\" font-size=\"28\">ALEX JORDAN</text>"
                           "<text x=\"40\" y=\"350\" font-size=\"18\">Nationality / Nasionaliteit</text>"
                           "<text x=\"40\" y=\"385\" font-size=\"28\">SOUTH AFRICAN</text>"
                           "<text x=\"40\" y=\"450\" font-size=\"18\">Identity No.</text>"
                           "<text x=\"40\" y=\"485\" font-size=\"28\">" + string{VALID} + "</text></g></svg>";
        const auto png = sc::svg2png::FromString(svg);
        if (png.size() < 3000) {
            cout << "   (no font to render text with: skipped)" << endl;
        } else {
            const auto path = (filesystem::temp_directory_path() / "sc-ai-test-passport-page.png").string();
            ofstream{path, ios::binary} << png;
            const sc::identity page{filesystem::path{path}};
            filesystem::remove(path);
            CHECK_EQ(page.surname(), string{"SAMPLESON"});
            CHECK_EQ(page.names(), string{"ALEX JORDAN"});
        }
    }

    TEST_SUMMARY();
}
