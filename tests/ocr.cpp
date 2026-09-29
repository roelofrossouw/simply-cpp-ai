#include "ocr.h"

#include <sc_test.h>
int main() {
    SECTION("Recognises text and returns positioned lines");
    {
        sc::ocr reader;
        reader.detect("resource/test/ID-10.jpg");

        CHECK(!reader.lines().empty());
        CHECK(!reader.text().empty());
        CHECK(reader.to_json().is_array());
        CHECK(reader.to_json().size() == reader.lines().size());

        for (const auto &line: reader.lines()) {
            CHECK(!line.text.empty());
            CHECK(line.confidence >= 0);
            CHECK(line.confidence <= 100);
            CHECK(line.box.width() > 0);
            CHECK(line.box.height() > 0);
        }

        reader.detect("resource/test/ID-10.jpg", 90);
        for (const auto &line: reader.lines()) CHECK(line.confidence >= 90);
    }

    SECTION("Rejects missing input");
    {
        sc::ocr reader;
        CHECK_THROWS_AS(reader.detect("resource/test/not-an-image.jpg"), std::runtime_error);
        CHECK_THROWS_AS(reader.detect("resource/test/ID-10.jpg", -1), std::invalid_argument);
        CHECK_THROWS_AS(reader.detect("resource/test/ID-10.jpg", 101), std::invalid_argument);
    }

    TEST_SUMMARY();
}
