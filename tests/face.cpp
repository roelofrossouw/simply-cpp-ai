#include <sc.h>
#include "facedetector.h"

using namespace std;
namespace fs = filesystem;

namespace {
    constexpr int MaxFiles = 100;
    constexpr size_t MaxFaces = 2;
    constexpr float MinFaceScore = 0.6f;
}

int main() {
    sc::timer sw;
    int counter{MaxFiles};
    sc::facedetector fd;
    fd.set_threshold(MinFaceScore);
    fd.set_max_faces(MaxFaces);
    vector<pair<string, pair<sc::face, sc::image> > > feats;

    for (const auto &img_file: fs::directory_iterator("resource/test/")) {
        if (!img_file.is_regular_file()) continue;
        if (!counter--) break;
        sc::timer sw2;
        fd.detect(img_file);
        if (!empty(fd.features())) feats.emplace_back(img_file.path().filename(), pair{fd.features().front(), fd.faces().front()});
        // if (!fd.display()) break; // Show annotated face.
    }
    cout << "\n\nDetection run: " << sw << endl;
    sw.reset();
    cout << "Comparing " << feats.size() << " x " << feats.size() << " = " << feats.size() * feats.size() << "\n";
    for (const auto &[n1, f1]: feats) {
        for (const auto &[n2, f2]: feats) {
            if (n1 >= n2) continue;
            auto sim = f1.first ^ f2.first;
            if (sim >= 40) {
                cout << (string) sim << " -> " << n1 << " vs " << n2;
                if (sim < 60) cout << "????";
                cout << endl;
                if (!f1.second.show(-1, "Image 1")) return 0;
                if (!f2.second.show(0, "Image 2")) return 0;
            }
        }
    }
    cout << "\n\nComparison run: " << sw << endl;
    return 0;
}
