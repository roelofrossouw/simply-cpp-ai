#include "identity.h"
#include <sc_test.h>

using namespace std;

namespace {
    constexpr auto VALID = "8001015009087";
    constexpr auto VALID_FEMALE = "9912310001083";
    constexpr auto BAD_CHECK_DIGIT = "8001015009086";
    constexpr auto BAD_MONTH = "8013015009082";
    constexpr auto BAD_CITIZENSHIP = "8001015009285";
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
        const sc::identity passport{"resource/test/PASPOORT-1.jpg"};
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

    TEST_SUMMARY();
}
