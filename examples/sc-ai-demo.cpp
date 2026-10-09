// Runs YOLO object detection (yolo26n.onnx from simply-cpp-models) on an image and lists what it
// found. Pass a photo to detect objects in it; without one it uses a small rendered image, which
// has no objects, but still shows the model, ONNX Runtime and sc-image loading and running.

#include <console.h>
#include <image.h>
#include <svg2png.h>
#include <timer.h>
#include <yolo.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    try {
        std::string source = argc > 1 ? argv[1] : "";
        sc::console::title("simply-cpp ai: object detection with YOLO");
        if (source.empty()) {
            sc::console::output() << "No photo given (sc-ai-demo <photo>), so using a rendered landscape with no objects in it\n";
            source = (std::filesystem::temp_directory_path() / "sc-ai-demo.png").string();
            std::ofstream{source, std::ios::binary} << sc::svg2png::FromString(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">)"
                R"(<rect width="640" height="480" fill="#87ceeb"/>)"
                R"(<rect y="360" width="640" height="120" fill="#3a7d44"/></svg>)");
        }
        sc::timer sw;

        // [readme]
        sc::console::heading("Loading the model");
        sc::yolo yolo{"yolo26n.onnx"};
        sc::console::show_text("sc::yolo yolo{\"yolo26n.onnx\"};", "loaded in " + std::string(sw));

        sc::console::heading("Detecting objects");
        sc::timer detecting;
        yolo.detect(std::filesystem::path{source});
        sc::console::show_text("yolo.detect(\"" + source + "\");", "done in " + std::string(detecting));

        // Each detection: centre x %, centre y %, width %, height %, confidence %, type.
        const auto detections = yolo.to_json();
        sc::console::heading(std::to_string(detections.size()) + (detections.size() == 1 ? " object found" : " objects found"));
        for (const auto &found: detections) {
            std::ostringstream line;
            line << std::left << std::setw(14) << sc::yolo::object_name(found[5].get<int>()) << std::right
                 << std::fixed << std::setprecision(1) << std::setw(5) << found[4].get<double>()
                 << "% sure, centred at (" << found[0].get<double>() << "%, " << found[1].get<double>() << "%), "
                 << found[2].get<double>() << "% x " << found[3].get<double>() << "% of the image";
            sc::console::note(line.str());
        }
        // [/readme]

        sc::console::output() << "\nAll of that took " << sw << ".\n";
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-demo: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
