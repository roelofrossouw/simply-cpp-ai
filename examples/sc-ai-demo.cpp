// Runs YOLO object detection (yolo26n.onnx from simply-cpp-models, or --model yolo26m) on an image, or on a video a
// frame at a time, and lists what it found. --help lists the options: a photo to detect objects
// in, or --video with a frame every so many seconds; without either it uses a small rendered image,
// which has no objects, but still shows the model, ONNX Runtime and sc-image loading and running.

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>

#include <nlohmann/json.hpp>

#include <params.h>
#include <sc.h>

#include <image.h>
#include <svg2png.h>
#include <video.h>
#include <yolo.h>

namespace {
    // A detection as a line: its type, how sure, where and how big (all in % of the image).
    std::string described(const nlohmann::ordered_json &found) {
        std::ostringstream line;
        line << std::left << std::setw(14) << sc::yolo::object_name(found[5].get<int>()) << std::right
             << std::fixed << std::setprecision(1) << std::setw(5) << found[4].get<double>()
             << "% sure, centred at (" << found[0].get<double>() << "%, " << found[1].get<double>() << "%), "
             << found[2].get<double>() << "% x " << found[3].get<double>() << "% of the image";
        return line.str();
    }

    // A time in the video as minutes, seconds and milliseconds: 01:02.500.
    std::string clock(const std::chrono::milliseconds time) {
        const auto ms = time.count();
        std::ostringstream out;
        out << std::setfill('0') << std::setw(2) << ms / 60000 << ':' << std::setw(2) << ms / 1000 % 60 << '.' << std::setw(3) << ms % 1000;
        return out.str();
    }
}

int main(int argc, char **argv) {
    // The command line: -h/--help comes with sc::params, which also reports mistakes and exits.
    const sc::params args{argc, argv, {
        {"name", "sc-ai-demo"},
        {"description", "Finds objects in a photo, or in a video a frame at a time, with YOLO and lists what it sees. "
                        "Without PHOTO or --video it runs on a rendered landscape with no objects in it, so you can "
                        "watch it work without a picture of your own."
                        "\nMore on the wiki: https://github.com/roelofrossouw/simply-cpp/wiki/AI"},
        {"options", {
            {"model", {{"short", "m"}, {"default", "yolo26n"}, {"value", "MODEL"},
                       {"help", "yolo26n (quick), yolo26m (slower, finds more and is surer), or an .onnx file"}}},
            {"video", {{"short", "v"}, {"value", "FILE"}, {"help", "a video to detect objects in (MP4, MOV, MKV, AVI, ...), instead of a photo"}}},
            {"every", {{"short", "e"}, {"type", "number"}, {"default", 1}, {"value", "SECONDS"},
                       {"help", "with --video, look at a frame this often (0: every frame)"}}},
            {"threshold", {{"short", "t"}, {"type", "number"}, {"value", "PERCENT"},
                           {"help", "list what it's at least this sure of, 0 to 100 (default: 30)"}}}}},
        {"positional", {
            {"photo", {{"required", false}, {"help", "the photo to detect objects in"}}}}}}};

    try {
        std::string source = args.as<std::string>("photo", "");
        const std::filesystem::path video_file = args.as<std::string>("video", "");
        if (!source.empty() && !video_file.empty()) throw std::invalid_argument("give a photo or --video, not both");
        const double every_seconds = args.as<double>("every");
        if (!(every_seconds >= 0)) throw std::invalid_argument("--every takes 0 seconds or more");
        const double threshold = args.as<double>("threshold", 30);
        if (!(threshold >= 0 && threshold <= 100)) throw std::invalid_argument("--threshold takes 0 to 100 (percent)");

        sc::console::title("simply-cpp ai: object detection with YOLO");
        if (source.empty() && video_file.empty()) {
            sc::console::output() << "No photo given (sc-ai-demo <photo>, or --video <file>), so using a rendered landscape with no objects in it\n";
            source = (std::filesystem::temp_directory_path() / "sc-ai-demo.png").string();
            std::ofstream{source, std::ios::binary} << sc::svg2png::FromString(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">)"
                R"(<rect width="640" height="480" fill="#87ceeb"/>)"
                R"(<rect y="360" width="640" height="120" fill="#3a7d44"/></svg>)");
        }
        sc::timer sw;

        // [readme]
        sc::console::heading("Loading the model");
        const auto model = args.as<std::string>("model"); // a name in simply-cpp-models, or a file
        sc::yolo yolo{model};
        sc::console::show_text("sc::yolo yolo{\"" + model + "\"};", "loaded in " + std::string(sw));
        yolo.set_threshold(threshold / 100); // a fraction: 0.3 lists what it's at least 30% sure of

        if (video_file.empty()) {
            sc::console::heading("Detecting objects");
            sc::timer detecting;
            yolo.detect(std::filesystem::path{source});
            sc::console::show_text("yolo.detect(\"" + source + "\");", "done in " + std::string(detecting));

            // Each detection: centre x %, centre y %, width %, height %, confidence %, type.
            const auto detections = yolo.to_json();
            sc::console::heading(std::to_string(detections.size()) + (detections.size() == 1 ? " object found" : " objects found"));
            for (const auto &found: detections) sc::console::note(described(found));
        } else {
            // A video: sc-image's sc::video reads it a frame at a time, each frame an sc::image.
            sc::console::heading("Detecting objects in a video");
            sc::video clip{video_file};
            std::ostringstream about;
            about << clip.size() << ", " << clip.fps() << " frames a second, " << clock(clip.duration());
            sc::console::show_text("sc::video clip{\"" + video_file.string() + "\"};", about.str());

            const std::chrono::milliseconds every{std::llround(every_seconds * 1000)};
            std::map<std::string, int> frames_with; // how many frames each kind of object was in
            int looked_at = 0;
            sc::timer detecting;
            while (const auto frame = clip.next(every)) { // the frames between are skipped, not decoded
                yolo.detect(frame->picture);
                ++looked_at;
                const auto detections = yolo.to_json();
                std::map<std::string, int> here;
                for (const auto &found: detections) ++here[sc::yolo::object_name(found[5].get<int>())];
                std::string line = clock(frame->time) + "  ";
                if (here.empty()) line += "nothing";
                for (const auto &[name, count]: here) {
                    line += (line.ends_with("  ") ? "" : ", ") + (count > 1 ? std::to_string(count) + " x " : "") + name;
                    ++frames_with[name];
                }
                sc::console::note(line);
            }
            sc::console::heading("In " + std::to_string(looked_at) + " frames, in " + std::string(detecting));
            if (frames_with.empty()) sc::console::note("nothing found");
            for (const auto &[name, count]: frames_with)
                sc::console::note(name + ": in " + std::to_string(count) + (count == 1 ? " frame" : " frames"));
        }
        // [/readme]

        sc::console::output() << "\nAll of that took " << sw << ".\n";
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-demo: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
