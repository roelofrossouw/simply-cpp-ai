#include <sc.h>
#include "onnx.h"
#include "image.h"
#include "ocr.h"

using namespace std;
using namespace sc;

#ifdef __APPLE__
const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};
#else
const filesystem::path image_path{"/var/www/build/sc-ai/tests/resource/test/"};
#endif

int main() {
    ocr ocr;
    for (const auto &file: filesystem::directory_iterator(image_path)) {
        if (file.path().filename() != "ID-10.jpg") continue;
        // if (file.path().filename() != "D1249 - MJ Mcameni - Paspoort 01.03.2031.jpg"
        //     && file.path().filename() != "ID-13.jpg"
        //     && file.path().filename() != "id-3.jpg")
        //     && file.path().filename() != "id-3.jpg")
        //     continue;
        if (!file.is_regular_file()) continue;
        timer sw;
        image img(file.path());
        cout << sw << " Loaded " << file.path().filename() << endl;
        sw.reset();
        ocr.detect(img);
        cout << sw << " Done OCR " << endl;
        sw.reset();
        cout << ocr.to_json().dump(4) << endl;
        cout << sw << " Done " << endl;
        if (!img.show()) break;
    }

    return 0;
}
