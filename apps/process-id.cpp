#include <sc.h>

#include "identity.h"

using namespace std;

const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};
const int MAXFILES = 5;

int main() {
    int files = MAXFILES;

    for (const auto &file: filesystem::directory_iterator(image_path)) {
        sc::timer sw;
        if (!file.is_regular_file()) continue;
        sc::identity id(file);
        cout << sw
                << " -> " << id.id_number()
                << ", " << id.passport_number()
                << ", " << id.names()
                << ", " << id.confidence()
                << " ==> " << file.path().filename()
                << endl;
        if (!--files) break;
    }
    return 0;
}
