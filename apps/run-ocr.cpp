#include <sc.h>
#include "ocr.h"
#include "image.h"

using namespace std;
const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};

int main() {
    sc::timer sw;
    sc::ocr reader;
    for (const auto &file: filesystem::directory_iterator(image_path)) {
        if (file.path().filename() != "ID-38b.jpg") continue;
        // if (file.path().filename() != "d1333 id.jpg") continue;
        if (!file.is_regular_file()) continue;
        sc::image img(file.path());
        reader.detect(img, 90);
        // img.rotate(180);
        // reader.detect(img, 90);
        cout << reader << endl;
        cout << "Took " << sw << endl;
        sw.reset();
    }

    return 0;
}
