#include <sc.h>
#include "facedetector.h"

using namespace std;
namespace fs = filesystem;

namespace {
    constexpr int MaxFiles = 10;
    constexpr size_t MaxFaces = 2;
    constexpr float MinFaceScore = 0.6f;
}

int main() {
    sc::timer sw;
    int counter{MaxFiles};
    sc::facedetector fd;
    fd.set_threshold(MinFaceScore);
    fd.set_max_faces(MaxFaces);
    vector<pair<string, sc::face> > feats;

    // Faces need real photographs: the private documents (resource/private/documents, kept out of
    // the repository). Without them there is nothing to detect.
    const fs::path documents{"resource/private/documents"};
    if (!fs::is_directory(documents)) {
        cout << "No private documents (" << documents.string() << "): skipped" << endl;
        return 0;
    }
    bool tested_image_input = false;
    for (const auto &img_file: fs::directory_iterator(documents)) {
        if (!img_file.is_regular_file()) continue;
        if (!counter--) break;
        sc::timer sw2;
        if (!tested_image_input) {
            const sc::image input{img_file.path().string()};
            const auto in_memory_faces = fd.detect(input);
            if (in_memory_faces.size() > MaxFaces) return 1;
            tested_image_input = true;
        }
        auto detected_faces = fd.detect(img_file.path());
        if (!detected_faces.empty()) feats.emplace_back(img_file.path().filename(), detected_faces.front());
        // if (!fd.display()) break; // Show annotated face.
    }
    cout << "\n\nDetection run: " << sw << endl;
    sw.reset();
    cout << "Comparing " << feats.size() << " x " << feats.size() << " = " << feats.size() * feats.size() << "\n";
    for (const auto &[n1, f1]: feats) {
        for (const auto &[n2, f2]: feats) {
            if (n1 >= n2) continue;
            auto sim = f1 ^ f2;
            if (sim >= 40) {
                cout << static_cast<string>(sim) << " -> " << n1 << " vs " << n2;
                if (sim < 60) cout << "????";
                cout << endl;
                // const auto matched_faces = sc::image::side_by_side(*f1.face_image(), *f2.face_image());
                // if (!matched_faces.show(0, "Matched faces")) return 0;
            }
        }
    }
    cout << "\n\nComparison run: " << sw << endl;
    return 0;
}
