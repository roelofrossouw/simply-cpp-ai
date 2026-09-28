#include "identity.h"
#include <algorithm>
#include <sc_test.h>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace std;

// Structurally valid numbers that belong to nobody: built by taking a birth
// date and sequence, then computing the Luhn digit, so the suite carries no
// real identity numbers.
namespace {
    constexpr auto VALID = "8001015009087"; // 1980-01-01, male, citizen
    constexpr auto VALID_FEMALE = "9912310001083"; // 1999-12-31, female, citizen
    constexpr auto BAD_CHECK_DIGIT = "8001015009086"; // VALID with the last digit wrong
    constexpr auto BAD_MONTH = "8013015009082"; // month 13; Luhn itself is correct
    constexpr auto BAD_CITIZENSHIP = "8001015009285"; // citizenship digit 2; Luhn is correct

    int count_readable(const vector<filesystem::path> &images, int &with_number) {
        with_number = 0;
        for (const auto &image: images) {
            const sc::identity document{image};
            if (document.has_id_number()) ++with_number;
        }
        return static_cast<int>(images.size());
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
        CHECK(!sc::identity::valid_id_number("800101500908")); // too short
        CHECK(!sc::identity::valid_id_number("80010150090855")); // too long
        CHECK(!sc::identity::valid_id_number("80010150090X5")); // not all digits
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

    SECTION("Unreadable input");
    {
        CHECK_THROWS_AS(sc::identity{"resource/test/does-not-exist.jpg"}, runtime_error);
    }

    SECTION("Reading a passport");
    {
        const sc::identity passport{"resource/test/PASPOORT-1.jpg"};
        CHECK(passport.has_id_number());
        CHECK(passport.document_type() == sc::identity::document::passport);
        CHECK(sc::identity::valid_id_number(passport.id_number()));
        CHECK(passport.checks().mrz);
        CHECK(passport.checks().luhn);
        CHECK_BETWEEN(static_cast<double>(passport.confidence()), 90.0, 100.0);
        CHECK(!passport.passport_number().empty());
        CHECK(!passport.date_of_expiry().empty());
        // The identity number's own digits must agree with the reported date.
        CHECK_EQ(passport.date_of_birth(), sc::identity::id_date_of_birth(passport.id_number()));
        CHECK_EQ(passport.sex(), sc::identity::id_sex(passport.id_number()));
    }

    SECTION("Reading a smart ID card");
    {
        const sc::identity card{"resource/test/ID-33.jpg"};
        CHECK(card.has_id_number());
        CHECK(card.document_type() == sc::identity::document::id_card);
        CHECK(sc::identity::valid_id_number(card.id_number()));
        CHECK(card.checks().date_of_birth); // printed date corroborates the number
        CHECK_BETWEEN(static_cast<double>(card.confidence()), 85.0, 100.0);
        CHECK(!card.surname().empty());
        CHECK(card.citizen().has_value());
    }

    SECTION("Optional face detection");
    {
        const sc::identity card{"resource/test/ID-33.jpg", true};
        CHECK(card.has_id_number());
        CHECK(!card.faces().empty());
        CHECK(card.faces().front().features().size() == sc::face::feature_size);
        CHECK(card.to_json().contains("faces"));

        const sc::identity without_faces{"resource/test/ID-33.jpg"};
        CHECK(without_faces.faces().empty());
        CHECK(!without_faces.to_json().contains("faces"));
    }

    SECTION("Reading a rotated document");
    {
        const sc::identity rotated{"resource/test/d1383 id.jpg"};
        CHECK(rotated.has_id_number());
        CHECK_EQ(rotated.rotation(), 0);
        CHECK(sc::identity::valid_id_number(rotated.id_number()));
    }

    SECTION("Reading a green ID book");
    {
        const sc::identity book{"resource/test/D1385 ID.jpg"};
        CHECK(book.has_id_number());
        CHECK(book.document_type() == sc::identity::document::id_book);
        CHECK(sc::identity::valid_id_number(book.id_number()));
    }

    SECTION("JSON output");
    {
        const sc::identity document{"resource/test/PASPOORT-1.jpg"};
        const auto json = document.to_json();
        CHECK_EQ(json["version"].get<int>(), 1);
        CHECK_EQ(json["document_type"].get<string>(), string{"passport"});
        CHECK_EQ(json["id_number"].get<string>(), document.id_number());
        CHECK(json["confidence"].is_number());
        CHECK(json["verification"]["luhn"].get<bool>());
        CHECK(json["verification"]["mrz"].get<bool>());
        CHECK(json.contains("rotation"));
        // The conversions agree with to_json().
        CHECK_EQ(static_cast<string>(document), json.dump());
        CHECK_EQ(document.to_json(), json);
    }

    SECTION("Reading the sample set");
    {
        vector<filesystem::path> images;
        const filesystem::path directory{"resource/test"};
        if (filesystem::exists(directory)) {
            for (const auto &entry: filesystem::directory_iterator{directory}) {
                if (entry.is_regular_file() && entry.path().extension() == ".jpg") images.push_back(entry.path());
            }
        }
        CHECK(images.size() > 20);

        ranges::sort(images);
        if (images.size() > 8) images.resize(8);
        int with_number = 0;
        const int total = count_readable(images, with_number);
        cout << "   read " << with_number << '/' << total << " sample documents" << endl;
        // Keep a small representative sample in the unit suite; broader OCR
        // accuracy sweeps are too expensive to run as part of every test pass.
        CHECK(with_number * 100 >= total * 80);
    }

    TEST_SUMMARY();
}
