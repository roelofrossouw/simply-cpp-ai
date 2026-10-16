#include "ocr.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sc_test.h>

int main() {
    SECTION("Recognises text and returns positioned lines");
    {
        // The specimen card is turned sideways: sc::ocr reads it upright, with boxes on the image
        // as given. Read once, as each read is a few seconds on a CPU.
        sc::ocr reader;
        const sc::image given{"resource/samples/card-sideways.jpg"};
        const int rotation = sc::ocr::upright_rotation(given);
        CHECK(rotation == 90 || rotation == 270);
        reader.detect(given, 0);
        CHECK_EQ(reader.rotation(), rotation);

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
            CHECK(line.box.left() >= -1 && line.box.right() <= given.size().width() + 1 &&
                  line.box.top() >= -1 && line.box.bottom() <= given.size().height() + 1);
        }
        const auto text = reader.text();
        for (const char *expected: {"REPUBLIC OF SOUTH AFRICA", "NATIONAL IDENTITY CARD", "SAMPLESON", "ALEX JORDAN"})
            CHECK_MSG(text.find(expected) != std::string::npos, std::string{"expected "} + expected);
    }

    SECTION("Detects regions and recognises extracted text");
    {
        // sc::ocr reads the card upright (it is turned sideways), so the reference does too.
        const sc::image given{"resource/samples/card-sideways.jpg"};
        sc::image input{given};
        const int rotation = sc::ocr::upright_rotation(given);
        CHECK(rotation == 90 || rotation == 270);
        if (rotation) input.rotate(rotation);
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

        // Recognition on a few of the lines: each one costs time on a CPU, and a few show it.
        const std::vector<sc::image> some(text_images.begin(), text_images.begin() + std::min<std::size_t>(8, text_images.size()));
        sc::ocr_recognizer recognizer;
        const auto recognised = recognizer.recognize(some);
        CHECK_EQ(recognised.size(), some.size());
        CHECK(std::ranges::any_of(recognised, [](const sc::ocr_recognizer::result &result) {
            return !result.text.empty();
        }));
        for (const auto &result: recognised) {
            CHECK(result.confidence >= 0);
            CHECK(result.confidence <= 100);
        }

        // The threshold leaves only the lines that reach it.
        constexpr double recognition_threshold = 90;
        const auto expected_count = std::ranges::count_if(recognised, [](const sc::ocr_recognizer::result &result) {
            return result.confidence >= recognition_threshold;
        });
        recognizer.set_threshold(recognition_threshold);
        const auto filtered = recognizer.recognize(some);
        CHECK_EQ(filtered.size(), expected_count);
        for (const auto &result: filtered) CHECK(result.confidence >= recognition_threshold);
        CHECK_THROWS_AS(recognizer.set_threshold(-1), std::invalid_argument);
        CHECK_THROWS_AS(recognizer.set_threshold(101), std::invalid_argument);
    }

    SECTION("Rejects missing input");
    {
        sc::ocr reader;
        CHECK_THROWS_AS(reader.detect("resource/samples/not-an-image.jpg"), std::runtime_error);
        CHECK_THROWS_AS(reader.detect("resource/samples/card-sideways.jpg", -1), std::invalid_argument);
        CHECK_THROWS_AS(reader.detect("resource/samples/card-sideways.jpg", 101), std::invalid_argument);
    }

    TEST_SUMMARY();
}
