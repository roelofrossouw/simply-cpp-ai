// Runs YOLO object detection (yolo26n.onnx from simply-cpp-models) on an image and lists what it
// found. Pass a photo to detect objects in it; without one it uses a small rendered image, which
// has no objects, but still shows the model, ONNX Runtime and sc-image loading and running.

#include <image.h>
#include <svg2png.h>
#include <timer.h>
#include <yolo.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {
    void heading(const std::string_view title) { std::cout << '\n' << title << '\n'; }
}

int main(int argc, char **argv) {
    try {
        std::string source = argc > 1 ? argv[1] : "";
        std::cout << "simply-cpp ai: object detection with YOLO\n";
        if (source.empty()) {
            std::cout << "No photo given (sc-ai-demo <photo>), so using a rendered landscape with no objects in it\n";
            source = (std::filesystem::temp_directory_path() / "sc-ai-demo.png").string();
            std::ofstream{source, std::ios::binary} << sc::svg2png::FromString(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">)"
                R"(<rect width="640" height="480" fill="#87ceeb"/>)"
                R"(<rect y="360" width="640" height="120" fill="#3a7d44"/></svg>)");
        }
        sc::timer sw;

        // [readme]
        heading("Loading the model");
        sc::yolo yolo{"yolo26n.onnx"};
        std::cout << "  sc::yolo yolo{\"yolo26n.onnx\"};\n      -> loaded in " << sw << '\n';

        heading("Detecting objects");
        sc::timer detecting;
        yolo.detect(std::filesystem::path{source});
        std::cout << "  yolo.detect(\"" << source << "\");\n      -> done in " << detecting << '\n';

        // Each detection: centre x %, centre y %, width %, height %, confidence %, type.
        const auto detections = yolo.to_json();
        heading(std::to_string(detections.size()) + (detections.size() == 1 ? " object found" : " objects found"));
        for (const auto &found: detections) {
            std::cout << "  " << std::left << std::setw(14) << sc::yolo::object_name(found[5].get<int>())
                      << std::right << std::fixed << std::setprecision(1) << std::setw(5) << found[4].get<double>()
                      << "% sure, centred at (" << found[0].get<double>() << "%, " << found[1].get<double>()
                      << "%), " << found[2].get<double>() << "% x " << found[3].get<double>() << "% of the image\n";
        }
        // [/readme]

        std::cout << "\nAll of that took " << sw << ".\n";
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-demo: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
