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
    // boxes(): where the faces are in the image's own pixels. A photo with a face (the COCO samples), and a copy
    // of half its size: the boxes come out half as big, at half the place, whatever the detector's letterbox.
    {
        sc::facedetector boxed;
        int checked = 0;
        for (const auto &file: fs::directory_iterator("resource/val2017")) {
            if (file.path().extension() != ".jpg") continue;
            const sc::image photo{file.path().string()};
            const auto faces = boxed.detect(photo, false);
            const auto boxes = boxed.boxes();
            if (boxes.size() != faces.size()) {
                cout << "FAILED: " << boxes.size() << " boxes for " << faces.size() << " faces in " << file.path() << endl;
                return 1;
            }
            if (faces.empty()) continue;
            const auto size = photo.size();
            const auto &box = boxes.front();
            if (box.left() < -box.width() / 2 || box.top() < -box.height() / 2 || box.right() > size.width() + box.width() / 2 ||
                box.bottom() > size.height() + box.height() / 2 || box.height() <= 0) {
                cout << "FAILED: a box outside its image: " << box.left() << "," << box.top() << " " << box.width() << "x" << box.height() << endl;
                return 1;
            }
            sc::image half{photo};
            half.resize_to({size.width() / 2, 0});
            const auto half_faces = boxed.detect(half, false);
            const auto half_boxes = boxed.boxes();
            if (half_faces.empty()) continue; // too small to find at half the size
            const double ratio = half_boxes.front().height() / box.height();
            const double moved = half_boxes.front().left() / std::max(1.0, box.left());
            cout << file.path().filename().string() << ": face " << box.width() << "x" << box.height() << " at " << box.left() << ","
                 << box.top() << "; half: x" << ratio << endl;
            if (ratio < 0.4 || ratio > 0.6 || (box.left() > 20 && (moved < 0.4 || moved > 0.6))) {
                cout << "FAILED: the half-size copy's box isn't half as big (" << ratio << ") or at half the place (" << moved << ")" << endl;
                return 1;
            }
            if (++checked == 3) break;
        }
        cout << checked << " photos' boxes checked" << endl;
        if (checked == 0) {
            cout << "FAILED: no face found in resource/val2017 to check boxes() with" << endl;
            return 1;
        }
    }
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
            }
        }
    }
    cout << "\n\nComparison run: " << sw << endl;
    return 0;
}
