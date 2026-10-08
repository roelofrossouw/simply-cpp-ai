// Runs YOLO object detection (yolo26n.onnx from simply-cpp-models) on an image file, or on a
// small rendered image when none is given - that finds nothing, but proves the model, ONNX
// Runtime and sc-image all load and run. Needs no server.

#include <image.h>
#include <svg2png.h>
#include <timer.h>
#include <yolo.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    try {
        std::string source = argc > 1 ? argv[1] : "";
        if (source.empty()) {
            source = (std::filesystem::temp_directory_path() / "sc-ai-demo.png").string();
            std::ofstream{source, std::ios::binary} << sc::svg2png::FromString(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">)"
                R"(<rect width="640" height="480" fill="#87ceeb"/>)"
                R"(<rect y="360" width="640" height="120" fill="#3a7d44"/></svg>)");
        }

        // [readme]
        sc::timer sw;
        const sc::yolo yolo{"yolo26n.onnx"};
        std::cout << "Model loaded after " << sw << '\n';

        yolo.detect(std::filesystem::path{source});
        std::cout << source << ": " << yolo << '\n';
        std::cout << "Done after " << sw << '\n';
        // [/readme]
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-demo: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
