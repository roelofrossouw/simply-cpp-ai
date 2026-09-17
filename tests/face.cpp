#include <sc.h>
#include "onnx.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

using namespace std;
namespace fs = filesystem;
const fs::path models("/opt/simply-cpp/models/");

namespace {
    constexpr int MaxFiles = 100;
    constexpr size_t MaxFaces = 2;
    constexpr float MinFaceScore = 0.6f;

    const auto DetModel = models / "det_10g.onnx";
    const auto EmbedModel = models / "w600k_r50.onnx";
    array STRIDES = {8, 16, 32};
    constexpr int stride = 32;
#ifdef __APPLE__
    // metal uses first size and makes it static, other engines should be able to use any size...
    std::vector<sc::size_i> valid_sizes = {{640, 640}};
#else
    std::vector<sc::size_i> valid_sizes = {
        {128, 128},
        {320, 320},
        {640, 640},
        {1024, 640},
        {640, 1024},
        {1024, 1024},
    };
#endif

    struct LayerInfo;
    struct Detection;

    const array<sc::point, 5> ARC_FACE_TEMPLATE = {
        {
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}
        }
    };

    sc::image extract_face(const Detection &detection, const sc::image &img);

    std::vector<sc::image> process_result(vector<const float *> outputs, sc::image &img);

    float *normalize(float *scores);

    float similarity(const float *a, const float *b);
}

std::ostream &operator<<(std::ostream &lhs, const pair<int, int> &rhs) {
    return lhs << "[" << rhs.first << " " << rhs.second << "]";
}

struct FaceFeatures {
    float data[512];
    FaceFeatures(float *f) { memcpy(data, f, sizeof(data)); }
    FaceFeatures(const FaceFeatures &f) { memcpy(data, f.data, sizeof(data)); }
};

int main() {
    sc::timer sw;
    int counter{MaxFiles};
    sc::onnx detect_face(DetModel);
    sc::onnx get_features(EmbedModel);
#ifndef NDEBUG
    sc::onnx::show_providers();
    detect_face.show_shapes();
    get_features.show_shapes();
#endif

    map<string, FaceFeatures> feats;
    for (const auto &img_file: fs::directory_iterator("resource/test/")) {
        if (!img_file.is_regular_file()) continue;
        if (!counter--) break;
        sc::image img(img_file.path());
        img.snap_to_size(stride, valid_sizes);
        auto result = detect_face.process_image(img);
        auto faces = process_result(result, img);
        if (faces.empty()) continue;
        int i = 0;
        for (auto &face: faces) {
            result = get_features.process_image(face);
            face.setFeatures(result[0]);
            normalize(face.getFeatures());
            feats.emplace(img_file.path().filename().string() + to_string(++i), face.getFeatures());
        }
    }
    cout << "\n\nDetection run: " << sw << endl;
    sw.reset();
    cout << "Comparing " << feats.size() << " x " << feats.size() << " = " << feats.size() * feats.size() << "\n";
    for (const auto &[n1, f1]: feats) {
        for (const auto &[n2, f2]: feats) {
            if (n1 == n2) continue;
            auto sim = similarity(f1.data, f2.data);
            if (sim >= 0.65) cout << "\n" << setw(15) << sim << "   " << n1 << " vs " << n2;
            else if (sim >= 0.40) cout << "\n" << setw(15) << sim << "   " << n1 << " ?? " << n2;
        }
        // cout << "\n";
    }
    cout << "\n\nComparison run: " << sw << endl;
    return 0;
}

namespace {
    struct LayerInfo {
        int stride{};
        sc::size_i feature_size{};

        LayerInfo() = default;

        LayerInfo(const int stride, const sc::size_i &size)
            : stride(stride), feature_size(size.width() / stride, size.height() / stride) {
        }

        [[nodiscard]] int count() const { return feature_size.width() * feature_size.height() * 2; }
    };

    struct Detection {
        int layer = -1;
        int index = -1;
        float score = 0.f;
        const float *lm = nullptr;
        int stride = 0;
        sc::size_i feature_size{0, 0};
        sc::point lt;
        sc::point rb;

        [[nodiscard]] int cell() const { return index / 2; }
        [[nodiscard]] int cx() const { return cell() % feature_size.width() * stride; }
        [[nodiscard]] int cy() const { return cell() / feature_size.height() * stride; }

        [[nodiscard]] array<sc::point, 5> landmarks() const {
            array<sc::point, 5> landmarks;
            for (int i = 0; i < 5; ++i)
                landmarks[i] = {
                    cx() + static_cast<int>(lm[i * 2] * static_cast<float>(stride)),
                    cy() + static_cast<int>(lm[i * 2 + 1] * static_cast<float>(stride))
                };
            return landmarks;
        }

        void calc_box(const vector<const float *> &outputs) {
            const float *box = outputs[3 + layer] + index * 4;
            const auto cx_ = cx();
            const auto cy_ = cy();
            auto x1 = cx_ - static_cast<int>(box[0] * static_cast<float>(stride));
            auto y1 = cy_ - static_cast<int>(box[1] * static_cast<float>(stride));
            auto x2 = cx_ + static_cast<int>(box[2] * static_cast<float>(stride));
            auto y2 = cy_ + static_cast<int>(box[3] * static_cast<float>(stride));
            lt = {x1, y1};
            rb = {x2, y2};
        }
    };


    bool overlaps(const Detection &lhs, const Detection &rhs) {
        return lhs.lt.x() < rhs.rb.x()
               && rhs.lt.x() < lhs.rb.x()
               && lhs.lt.y() < rhs.rb.y()
               && rhs.lt.y() < lhs.rb.y();
    }

    void get_face_box(const Detection &det, const sc::image &img) {
        img.rect(det.lt, det.rb);
        const sc::point_i label_position{det.lt.x(), std::max(20, (int) det.lt.y() - 5)};
        const std::string label = to_string(static_cast<int>(det.score * 100)) + "%";
        img.text(label, label_position);
    }

    sc::image extract_face(const Detection &detection, const sc::image &img) {
        return img.warp(detection.landmarks(), ARC_FACE_TEMPLATE, {112, 112});;
    }

    void annotate_face(const Detection &detection, const sc::image &img) {
        get_face_box(detection, img);
        for (const auto &l: detection.landmarks()) img.circle({l.x(), l.y()}, 1);
    }

    vector<sc::image> process_result(vector<const float *> outputs, sc::image &img) {
        std::vector<LayerInfo> layers;
        layers.reserve(STRIDES.size());
        for (const int stride_size: STRIDES) layers.emplace_back(stride_size, img.size());

        std::vector<Detection> candidates;
        for (size_t layer = 0; layer < layers.size(); ++layer) {
            auto scores = outputs[layer];
            for (int index = 0; index < layers[layer].count(); ++index) {
                if (scores[index] < MinFaceScore) continue;
                Detection detection;
                detection.layer = static_cast<int>(layer);
                detection.index = index;
                detection.score = scores[index];
                detection.stride = layers[layer].stride;
                detection.feature_size = layers[layer].feature_size;
                detection.lm = outputs[6 + layer] + index * 10;
                detection.calc_box(outputs);
                candidates.push_back(detection);
            }
        }

        ranges::sort(candidates, [](const Detection &lhs, const Detection &rhs) {
            return lhs.score > rhs.score;
        });

        std::vector<Detection> selected;
        selected.reserve(std::min(MaxFaces, candidates.size()));
        for (const auto &candidate: candidates) {
            if (ranges::any_of(selected, [&candidate](const Detection &selected_detection) {
                return overlaps(candidate, selected_detection);
            }))
                continue;

            selected.push_back(candidate);
            if (selected.size() == MaxFaces) break;
        }

        std::vector<sc::image> faces;
        faces.reserve(selected.size());

        for (const auto &detection: selected) faces.push_back(extract_face(detection, img));
        for (const auto &detection: selected) annotate_face(detection, img);
        return faces;
    }

    float *normalize(float *scores) {
        float norm = 0.0f;
        for (size_t i = 0; i < 512; ++i) norm += scores[i] * scores[i];
        norm = std::sqrt(norm);
        for (size_t i = 0; i < 512; ++i) scores[i] /= norm;
        return scores;
    }

    float similarity(const float *a, const float *b) {
        return std::inner_product(a, a + 512, b, 0.0f);
        // same image      1.00
        // same person     0.65 - 0.90
        // different       0.10 - 0.40

        // float dot = 0.f;
        // for (int i = 0; i < 512; ++i) dot += a[i] * b[i];
        // return dot;
    }
}
