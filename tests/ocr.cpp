#include "ocr.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <svg2png.h>
#include <sc_test.h>

namespace {
    // A white page, width pixels wide, with three lines of black text scaled to suit.
    std::string invoice_page(const int width, const bool with_text = true) {
        const int font_size = width * 24 / 640;
        std::string svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + std::to_string(width) +
                          "\" height=\"" + std::to_string(font_size * 8) + "\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>";
        if (with_text) {
            svg += "<g font-family=\"Arial, Helvetica, DejaVu Sans, Liberation Sans, sans-serif\" font-size=\"" +
                    std::to_string(font_size) + "\">";
            const char *lines[] = {"Invoice 2026-0142", "Total due: R 1 250.00", "Due date: 2026-10-31"};
            for (int i = 0; i < 3; ++i)
                svg += "<text x=\"" + std::to_string(font_size) + "\" y=\"" + std::to_string(font_size * 2 * (i + 1)) +
                        "\">" + lines[i] + "</text>";
            svg += "</g>";
        }
        return svg + "</svg>";
    }
}

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

        const sc::image input{"resource/test/ID-10.jpg"};
        sc::ocr_detector detector;
        sc::ocr_recognizer recognizer;
        const auto regions = detector.detect(input);
        const auto text_images = detector.text_images(input, regions);
        std::vector<sc::image> orientation_images;
        orientation_images.reserve(text_images.size() * 2);
        for (const auto &text_image: text_images) {
            orientation_images.push_back(text_image);
            auto &upside_down = orientation_images.emplace_back(text_image);
            upside_down.rotate(180);
        }
        const auto recognized = recognizer.recognize(orientation_images);
        reader.detect(input, 0);
        std::vector<sc::ocr_recognizer::result> expected;
        for (size_t i = 0; i < regions.size(); ++i) {
            const auto &forward = recognized[i * 2];
            const auto &upside_down = recognized[i * 2 + 1];
            const auto &best = upside_down.confidence > forward.confidence ? upside_down : forward;
            if (!best.text.empty()) expected.push_back(best);
        }
        CHECK_EQ(reader.lines().size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK_EQ(reader.lines()[i].text, expected[i].text);
            CHECK_EQ(reader.lines()[i].confidence, expected[i].confidence);
        }
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

        constexpr double recognition_threshold = 90;
        const auto expected_count = std::ranges::count_if(recognised, [](const sc::ocr_recognizer::result &result) {
            return result.confidence >= recognition_threshold;
        });
        recognizer.set_threshold(recognition_threshold);
        const auto filtered = recognizer.recognize(text_images);
        CHECK_EQ(filtered.size(), expected_count);
        for (const auto &result: filtered) CHECK(result.confidence >= recognition_threshold);
        CHECK_THROWS_AS(recognizer.set_threshold(-1), std::invalid_argument);
        CHECK_THROWS_AS(recognizer.set_threshold(101), std::invalid_argument);
    }

    SECTION("Reads a few lines of clean text whole and in reading order, at any size");
    {
        // A page with fewer text regions than the noise filter's neighbour count used to come back
        // empty, and large text came back word by word in a shuffled order.
        sc::ocr reader;
        const auto path = (std::filesystem::temp_directory_path() / "sc-ai-test-ocr-page.png").string();
        for (const int width: {480, 640, 1600}) {
            const auto png = sc::svg2png::FromString(invoice_page(width));
            // Rendering text needs a system font; without one the page stays blank.
            if (png.size() < sc::svg2png::FromString(invoice_page(width, false)).size() + 1000) {
                std::cout << "   (no font to render text with: skipped)" << std::endl;
                break;
            }
            std::ofstream{path, std::ios::binary} << png;
            reader.detect(std::filesystem::path{path});
            CHECK_MSG(reader.text() == "Invoice 2026-0142\nTotal due: R 1 250.00\nDue date: 2026-10-31",
                      "at " + std::to_string(width) + " px, read \"" + reader.text() + '"');
        }
        std::filesystem::remove(path);
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
