#include <sc.h>
#include "onnx.h"
#include "image.h"
#include "ocr.h"

using namespace std;
using namespace sc;

const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};

int main() {
    ocr_detector det;
    ocr_recognizer rec;
    // onnx rotate("paddle_line_rotate");
    // rotate.show_shapes();

    for (const auto &file: filesystem::directory_iterator(image_path)) {
        if (file.path().filename() != "D1249 - MJ Mcameni - Paspoort 01.03.2031.jpg"
            && file.path().filename() != "ID-13.jpg"
            && file.path().filename() != "id-3.jpg")
            continue;
        if (!file.is_regular_file()) continue;
        timer sw;
        image img(file.path());
        cout << sw << " Loaded " << file.path().filename() << endl;
        sw.reset();
        auto rects = det.detect(img);
        cout << sw << " Got rects " << endl;
        sw.reset();
        // //auto dets = det.text_images(img, rects);
        std::vector<image> result;
        result.reserve(rects.size() * 2);

        for (auto &region: rects) {
            auto image = img.deskewed(region);
            result.push_back(image);
            image.rotate(180);
            result.push_back(image);
        }
        cout << sw << " Got images " << endl;
        sw.reset();
        auto txt = rec.recognize(result);
        cout << sw << " Got inference " << endl;
        sw.reset();

        //     if (region.left() < 5) continue; // Probable cut off words
        //     auto txt_img = img.deskewed(region);
        //     auto sample = txt_img.resized({0, 80});
        //     auto txt = rec.recognize(txt_img);
        //     if (txt.confidence < 80) {
        //         txt_img.rotate(180);
        //         auto txt2 = rec.recognize(txt_img);
        //         if (txt2.confidence > txt.confidence) {
        //             if (txt2.confidence < 80) cout << "Bad both ways..." << endl;
        //             else cout << "Rotated" << " - " << txt2.confidence << ": " << txt2.text << endl;
        //         } else {
        //             cout << "Bad confidence" << " - " << txt.confidence << ": " << txt.text << endl;
        //         }
        //     } else {
        //         cout <<  " - " << txt.confidence << ": " << txt.text << endl;
        //     }
        //
        //     // if (!txt_img.show()) break;
        // }
        cout << sw << " Done " << endl;
    }

    return 0;
}
