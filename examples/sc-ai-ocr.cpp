// Reads text from an image with PaddleOCR models (from simply-cpp-models): finds each line, reads
// it, and gives how sure it is and where it is. Pass an image to read that; without one it renders
// a few lines of text from SVG and reads those. Writes only to the temporary directory.

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <sc.h>

#include <image.h>
#include <ocr.h>
#include <svg2png.h>

int main(int argc, char **argv) {
    try {
        std::string source = argc > 1 ? argv[1] : "";
        sc::console::title("simply-cpp ai: reading text with OCR");
        if (source.empty()) {
            sc::console::output() << "No image given (sc-ai-ocr <image>), so rendering a few lines of text\n";
            source = (std::filesystem::temp_directory_path() / "sc-ai-ocr.png").string();
            std::ofstream{source, std::ios::binary} << sc::svg2png::FromString(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220">)"
                R"(<rect width="640" height="220" fill="white"/>)"
                R"(<g font-family="Arial, Helvetica, DejaVu Sans, sans-serif" font-size="30">)"
                R"(<text x="30" y="60">Invoice 2026-0142</text>)"
                R"(<text x="30" y="120">Total due: R 1 250.00</text>)"
                R"(<text x="30" y="180">Due date: 2026-10-31</text></g></svg>)");
        }
        sc::timer sw;

        sc::console::heading("Loading the models");
        sc::ocr reader;
        sc::console::show_text("sc::ocr reader;", "text detection and recognition models loaded in " + std::string(sw));

        sc::console::heading("Reading");
        sc::timer reading;
        reader.detect(std::filesystem::path{source});
        sc::console::show_text("reader.detect(\"" + source + "\");", "done in " + std::string(reading));

        sc::console::heading(std::to_string(reader.lines().size()) + (reader.lines().size() == 1 ? " line" : " lines"));
        for (const auto &line: reader.lines())
            sc::console::note(sc::console::format(line.text) + "  " + std::string(line.confidence) + " sure, at " +
                              sc::console::format(line.box));

        sc::console::subheading("All the text at once");
        sc::console::show_text("reader.text()", reader.text());
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-ocr: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
