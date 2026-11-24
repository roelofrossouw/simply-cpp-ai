#include <sc/core/sc.h>
#include <sc/ai/onnx.h>
#include <sc/ai/ocr.h>
#include <sc/ai/yolo.h>
#include <sc/image/video.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include "sc_test.h"

using namespace std;
namespace fs = filesystem;

const fs::path IMAGES = "resource/val2017/";
const fs::path MODEL = "yolo26n.onnx";
constexpr int MaxFiles = 100;

int main() {
    SECTION("An output's size is the product of its shape");
    {
        std::vector<float> values(1 * 4 * 6, 0.f);
        values[17] = 9.f;
        const sc::ai::output out{{1, 4, 6}, values.data()};
        CHECK_EQ(out.data_size(), 24);   // 1 * 4 * 6, not 1 + 4 + 6
        CHECK_EQ(out.max_element(), 17); // so the search reaches the end
        CHECK_EQ((sc::ai::output{{}, values.data()}.data_size()), 0);
    }

    SECTION("A model's image size is width by height");
    {
        const sc::ai::onnx detector{MODEL.string()};
        CHECK_EQ(detector.image_size().width(), 640);
        CHECK_EQ(detector.image_size().height(), 640);
        // The line orientation model takes 160 x 80: [batch, 3, 80, 160].
        const sc::ai::onnx lines{"paddle_line_rotate"};
        CHECK_EQ(lines.image_size().width(), 160);
        CHECK_EQ(lines.image_size().height(), 80);
        // The text recogniser takes lines 48 high, of any width: [batch, 3, 48, -1].
        const sc::ai::onnx recogniser{RECOGNITION_MODEL};
        CHECK_EQ(recogniser.image_size().height(), 48);
        CHECK_LT(recogniser.image_size().width(), 1); // dynamic
    }

    SECTION("Models run on the GPU asked for; one that isn't there is refused");
    {
        const int gpus = sc::ai::onnx::gpus();
        cout << gpus << (gpus == 1 ? " GPU" : " GPUs") << endl;
        CHECK(gpus >= 0);
        if (gpus > 0) {
            const sc::ai::yolo last{MODEL.string(), gpus - 1}; // each GPU takes a model
            CHECK_THROWS_AS(sc::ai::yolo(MODEL.string(), gpus), std::invalid_argument);
            CHECK_THROWS_AS(sc::ai::yolo(MODEL.string(), -2), std::invalid_argument);
            CHECK(last.on_gpu());
        }
    }

    SECTION("onnx::cpu puts a model on the CPU, whatever GPU there is");
    {
        const sc::ai::yolo on_cpu{MODEL.string(), sc::ai::onnx::cpu};
        CHECK(!on_cpu.on_gpu());
        const sc::ai::onnx raw{MODEL.string(), true, sc::ai::onnx::cpu};
        CHECK(!raw.on_gpu());
        // The default: a GPU where there is one (CoreML on Apple platforms), else the CPU too.
        const sc::ai::onnx default_device{MODEL.string()};
#ifdef __APPLE__
        CHECK(default_device.on_gpu());
#else
        CHECK_EQ(default_device.on_gpu(), sc::ai::onnx::gpus() > 0);
#endif
        // It detects all the same.
        on_cpu.detect(sc::image::picture{(IMAGES / "000000031296.jpg").string()}); // people at a table
        CHECK(!static_cast<nlohmann::ordered_json>(on_cpu).empty());
    }

    SECTION("Nothing found is an empty list, not null");
    {
        const std::vector<float> grey(3 * 320 * 320, 0.5f);
        const auto blank = sc::image::picture::from_blob(grey.data(), 320, 320);
        const sc::ai::yolo nothing{MODEL.string()};
        nothing.detect(blank);
        CHECK(nothing.to_json().is_array());
        CHECK(nothing.to_json().empty());
        CHECK_EQ(static_cast<std::string>(nothing), std::string{"[]"});
    }

    SECTION("The medium model: the same output, surer and finding more");
    {
        const fs::path medium = fs::path{SC_AI_PACKAGE_MODELS} / "yolo26m.onnx";
        const sc::ai::onnx detector{medium.string()};
        CHECK_EQ(detector.image_size().width(), 640);
        CHECK_EQ(detector.yolo26_size(), 10); // exported with max_det=10, as the nano model was
        const auto people = [](const sc::ai::yolo &model, double &best) {
            int count = 0;
            for (const auto &found: model.to_json())
                if (sc::ai::yolo::object_name(found[5].get<int>()) == "person") {
                    ++count;
                    best = std::max(best, found[4].get<double>());
                }
            return count;
        };
        const fs::path picture = IMAGES / "000000031296.jpg"; // people at a table
        sc::ai::yolo nano{MODEL.string()}, more{medium.string()};
        nano.detect(picture);
        more.detect(picture);
        double nano_best = 0, medium_best = 0;
        const int nano_people = people(nano, nano_best), medium_people = people(more, medium_best);
        CHECK(medium_people >= nano_people);
        CHECK(medium_best >= 90);
        CHECK(nano_people > 0);
    }

    SECTION("A video: each kind's surest detection, as for an image; or, in full, a summary");
    {
        // Two seconds at 10 frames a second: one of grey, then one of people at a table.
        const auto clip_file = fs::temp_directory_path() / "sc-ai-yolo-test.avi";
        {
            cv::Mat people = cv::imread((IMAGES / "000000031296.jpg").string());
            cv::resize(people, people, cv::Size{640, 427});
            cv::VideoWriter writer{clip_file.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, people.size()};
            CHECK(writer.isOpened());
            for (int n = 0; n < 10; ++n) writer.write(cv::Mat{people.size(), CV_8UC3, cv::Scalar::all(128)});
            for (int n = 0; n < 10; ++n) writer.write(people);
        }
        const sc::ai::yolo nano{MODEL.string()};
        sc::image::video clip{clip_file};
        const auto summary = nano.detect(clip, std::chrono::milliseconds{0}, sc::ai::yolo::detail::full); // every frame
        CHECK_EQ(summary["frames"].get<int>(), 20);
        CHECK_EQ(summary["every"].get<int>(), 0);
        const auto &person = summary["objects"]["person"];
        CHECK_EQ(person["count"].get<int>(), 10);           // in the second second only
        CHECK(person["frame"].get<int>() >= 10);
        CHECK_EQ(person["time"].get<int>(), person["frame"].get<int>() * 100);
        CHECK(person["confidence"].get<double>() >= 80);
        bool listed = false; // the surest frame's own detections include that best person
        for (const auto &found: person["detections"])
            listed = listed || (found[5].get<int>() == 0 && found[4].get<double>() == person["confidence"].get<double>());
        CHECK(listed);
        double previous = 101; // most certain first
        for (const auto &[name, kind]: summary["objects"].items()) {
            CHECK(kind["confidence"].get<double>() <= previous);
            previous = kind["confidence"].get<double>();
        }
        // Without detail::full, what an image gives: each kind's surest detection, most certain first.
        sc::image::video again{clip_file};
        const auto best = nano.detect(again, std::chrono::milliseconds{0});
        CHECK(best.is_array());
        CHECK_EQ(best.size(), summary["objects"].size()); // one a kind
        CHECK_EQ(best.at(0).size(), std::size_t{6});
        CHECK_EQ(best.at(0)[5].get<int>(), 0); // the person is the surest
        CHECK_EQ(best.at(0)[4].get<double>(), person["confidence"].get<double>());
        for (std::size_t i = 1; i < best.size(); ++i) CHECK(best.at(i)[4].get<double>() <= best.at(i - 1)[4].get<double>());

        // detect_until(): stop() is asked before each frame; once it says true, nothing is returned.
        sc::image::video stopped{clip_file};
        int asked = 0;
        const auto cut_short = nano.detect_until(stopped, [&asked] { return ++asked > 3; }, std::chrono::milliseconds{0});
        CHECK(!cut_short.has_value());
        CHECK_EQ(asked, 4); // three frames run, then it stopped
        sc::image::video whole{clip_file};
        const auto never = nano.detect_until(whole, [] { return false; }, std::chrono::milliseconds{0});
        CHECK(never.has_value());
        CHECK_EQ(*never, best); // as detect()

        sc::image::video grey_only{clip_file};
        sc::ai::yolo blind{MODEL.string()};
        blind.set_threshold(100);
        const auto none = blind.detect(grey_only, std::chrono::seconds(1));
        CHECK(none.is_array() && none.empty()); // [] when nothing's found, as for an image
        fs::remove(clip_file);
    }

    sc::timer sw;
    sc::ai::yolo yolo(MODEL.string());
    yolo.set_threshold(50);
    int counter{MaxFiles};

    cout << "Running max " << counter << " files." << endl;
    for (const auto &img_file: fs::directory_iterator(IMAGES)) {
        if (!img_file.is_regular_file()) continue;
        if (img_file.path().filename().string().starts_with('.')) continue;
        if (!counter--) break;
        sc::timer image_timer;
        yolo.detect(img_file.path());
        cout << img_file.path().filename() << ": " << yolo << " (" << image_timer << ")" << endl;
// #ifdef __APPLE__
// #endif
    }

    cout << "\n\nTotal run: " << sw << endl;
    TEST_SUMMARY();
}
