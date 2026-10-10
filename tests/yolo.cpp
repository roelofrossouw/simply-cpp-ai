#include <sc.h>
#include <onnx.h>
#include <ocr.h>
#include <yolo.h>
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
        const sc::output out{{1, 4, 6}, values.data()};
        CHECK_EQ(out.data_size(), 24);   // 1 * 4 * 6, not 1 + 4 + 6
        CHECK_EQ(out.max_element(), 17); // so the search reaches the end
        CHECK_EQ((sc::output{{}, values.data()}.data_size()), 0);
    }

    SECTION("A model's image size is width by height");
    {
        const sc::onnx detector{MODEL.string()};
        CHECK_EQ(detector.image_size().width(), 640);
        CHECK_EQ(detector.image_size().height(), 640);
        // The line orientation model takes 160 x 80: [batch, 3, 80, 160].
        const sc::onnx lines{"paddle_line_rotate"};
        CHECK_EQ(lines.image_size().width(), 160);
        CHECK_EQ(lines.image_size().height(), 80);
        // The text recogniser takes lines 48 high, of any width: [batch, 3, 48, -1].
        const sc::onnx recogniser{RECOGNITION_MODEL};
        CHECK_EQ(recogniser.image_size().height(), 48);
        CHECK_LT(recogniser.image_size().width(), 1); // dynamic
    }

    SECTION("Nothing found is an empty list, not null");
    {
        const std::vector<float> grey(3 * 320 * 320, 0.5f);
        const auto blank = sc::image::from_blob(grey.data(), 320, 320);
        const sc::yolo nothing{MODEL.string()};
        nothing.detect(blank);
        CHECK(nothing.to_json().is_array());
        CHECK(nothing.to_json().empty());
        CHECK_EQ(static_cast<std::string>(nothing), std::string{"[]"});
    }

    sc::timer sw;
    sc::yolo yolo(MODEL.string());
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
//         if (!yolo.display(0)) break;
// #endif
    }

    cout << "\n\nTotal run: " << sw << endl;
    TEST_SUMMARY();
}
