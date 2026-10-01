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

    SECTION("Detects regions and recognises extracted text");
    {
        const sc::image input{"resource/test/ID-10.jpg"};
        sc::ocr_detector detector;
        const auto regions = detector.detect(input);
        CHECK(!regions.empty());
        for (const auto &region: regions) {
            CHECK(region.width() > 0);
            CHECK(region.height() > 0);
        }

        const auto text_images = detector.text_images(input, regions);
        CHECK_EQ(text_images.size(), regions.size());
        for (const auto &text_image: text_images) CHECK(!text_image.empty());

        const auto detected_images = detector.text_images(input);
        CHECK_EQ(detected_images.size(), regions.size());

        sc::ocr_recognizer recognizer;
        const auto recognised = recognizer.recognize(text_images);
        CHECK_EQ(recognised.size(), text_images.size());
        CHECK(std::ranges::any_of(recognised, [](const sc::ocr_recognizer::result &result) {
            return !result.text.empty();
        }));
        for (const auto &result: recognised) {
            CHECK(result.confidence >= 0);
            CHECK(result.confidence <= 100);
        }
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
