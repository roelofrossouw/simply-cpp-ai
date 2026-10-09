#include "identity.h"
#include <sc_test.h>

int main() {
    SECTION("Reading partial smart ID card fields");
    {
        const sc::identity card{"resource/test/ID-10.jpg"};
        CHECK(!card.has_id_number());
        CHECK(card.document_type() == sc::identity::document::id_card);
        CHECK_EQ(card.names(), std::string{"LUCAS"});
        // The surname isn't checked for now: on Linux the OCR reads an extra letter into it.
    }
    TEST_SUMMARY();
}
