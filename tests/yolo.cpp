#include <sc.h>
#include <yolo.h>
#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = filesystem;

const fs::path IMAGES = "resource/val2017/";
const fs::path MODEL = "yolo26n.onnx";
constexpr int MaxFiles = 100;

int main() {
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
    return 0;
}
