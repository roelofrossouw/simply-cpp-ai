#include "facedetector.h"
#include "onnx.h"
#include <stdexcept>

using namespace std;

namespace sc {
    namespace impl {
        static array STRIDES = {8, 16, 32};
        constexpr int stride = 32;
#ifdef __APPLE__
        // metal uses first size and makes it static, other engines should be able to use any size...
        vector<size_i> valid_sizes = {{640, 640}};
#else
        vector<size_i> valid_sizes = {
            {128, 128},
            {320, 320},
            {640, 640},
            {1024, 640},
            {640, 1024},
            {1024, 1024},
        };
#endif
        const array<point, 5> ARC_FACE_TEMPLATE = {
            {
                {38.2946f, 51.6963f},
                {73.5318f, 51.5014f},
                {56.0252f, 71.7366f},
                {41.5493f, 92.3655f},
                {70.7299f, 92.2041f}
            }
        };

        struct FaceFeatures {
            float data[512]{};
            explicit FaceFeatures(float *f) { memcpy(data, f, sizeof(data)); }
            FaceFeatures(const FaceFeatures &f) { memcpy(data, f.data, sizeof(data)); }
        };

        struct Detection {
            int layer = -1;
            int index = -1;
            percent score = {0.f, 0};
            const float *lm = nullptr;
            int stride = 0;
            size_i feature_size{0, 0};
            rect box;

            [[nodiscard]] int cell() const { return index / 2; }
            [[nodiscard]] point center() const { return {cx(), cy()}; }
            [[nodiscard]] int cx() const { return cell() % feature_size.width() * stride; }
            [[nodiscard]] int cy() const { return cell() / feature_size.height() * stride; }

            [[nodiscard]] array<point, 5> landmarks() const {
                array<point, 5> landmarks;
                for (int i = 0; i < 5; ++i) landmarks[i] = center() + point{lm[i * 2], lm[i * 2 + 1]} * stride;
                return landmarks;
            }

            void calc_box(const vector<const float *> &outputs) {
                const float *c = outputs[3 + layer] + index * 4;
                box = rect::ltrb(-c[0], -c[1], c[2], c[3]) * stride + center();
            }

            [[nodiscard]] bool overlaps(const Detection &rhs) const { return box.iou(rhs.box) > 0.2; }
        };

        struct LayerInfo {
            int stride{};
            size_i feature_size{};

            LayerInfo() = default;

            LayerInfo(const int stride, const size_i &size)
                : stride(stride),
                  feature_size(size.width() / stride, size.height() / stride) {
            }

            [[nodiscard]] int count() const {
                return feature_size.width() * feature_size.height() * 2;
            }
        };

        class face_impl {
        public:
            explicit face_impl() : detector("det_10g.onnx"),
                                   extractor("w600k_r50.onnx") {
#ifndef NDEBUG
                onnx::show_providers();
                detector.show_shapes();
                extractor.show_shapes();
#endif
            }

            void set_threshold(const float threshold) {
                threshold_ = threshold;
            }

            void set_max_faces(const int max_faces) {
                max_faces_ = max_faces;
            }

            static image extract_face(const Detection &detection, const image &img) {
                return img.warp(detection.landmarks(), ARC_FACE_TEMPLATE, {112, 112});;
            }

            static void annotate_face(const Detection &detection, image &img) {
                img.rect(detection.box);
                const point_i label_position{detection.box.left(), std::max(20, (int) detection.box.top() - 5)};
                const std::string label = detection.score;
                img.text(label, label_position);
                for (const auto &l: detection.landmarks()) img.circle({l.x(), l.y()}, 1);
            }

            void process_detections(const vector<const float *> &outputs, const image &img) {
                vector<LayerInfo> layers;
                layers.reserve(STRIDES.size());
                const auto image_size = img.size();
                for (const int stride_size: STRIDES) layers.emplace_back(stride_size, image_size);

                vector<Detection> candidates;
                for (size_t layer = 0; layer < layers.size(); ++layer) {
                    auto scores = outputs[layer];
                    for (int index = 0; index < layers[layer].count(); ++index) {
                        if (scores[index] < threshold_) continue;
                        Detection detection;
                        detection.layer = static_cast<int>(layer);
                        detection.index = index;
                        detection.score = {scores[index], 0};
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

                for (const auto &candidate: candidates) {
                    const auto overlaps = ranges::any_of(detections, [&candidate](const Detection &selected_detection) {
                        return candidate.overlaps(selected_detection);
                    });
                    if (overlaps) continue;
                    detections.push_back(candidate);
                    if (detections.size() == max_faces_) break;
                }
                for (const auto &detection: detections) faces.emplace_back(extract_face(detection, img));
            }

            void run(const string &image_filename) {
                detections.clear();
                faces.clear();
                features.clear();
                original = make_unique<image>(image_filename);
                image input{*original};
                input.snap_to_size(stride, valid_sizes);
                auto result = detector.process_image(input);
                process_detections(result, input);
                if (faces.empty()) return;
                for (auto &f: faces) {
                    features.emplace_back(extractor.process_image(f)[0]);
                }
            }

            [[nodiscard]] nlohmann::ordered_json json() const {
                nlohmann::ordered_json result;
                // for (const auto &detection: detections)
                //     if (detection.confidence >= threshold_) result.push_back(detection.json());
                return result;
            }

            [[nodiscard]] image annotated() const {
                if (!original) throw runtime_error("No image has been detected");
                image result{*original};
                result.snap_to_size(stride, valid_sizes);
                // for (const auto &face: faces) if (!face.show()) break;
                for (const auto &detection: detections) annotate_face(detection, result);
                return result;
            }

            onnx detector; // Model to locate the face
            onnx extractor; // Model to extract features
            vector<Detection> detections;
            unique_ptr<image> original;
            vector<image> faces;
            vector<face> features;

        private:
            float threshold_{0.6f};
            size_t max_faces_{2};
        };
    }

    face::face(const float *feats) {
        setFeatures(feats);
    }

    face::face(const face &copy) {
        setFeatures(copy.features);
    }

    void face::setFeatures(const float *feats) {
        std::memcpy(features, feats, sizeof(features));
        normalize();
    }

    percent face::similarity(const face &rhs) const {
        return {std::inner_product(features, features + 512, rhs.features, 0.0f), 0};
    }

    percent face::operator^(const face &rhs) const {
        return similarity(rhs);
    }

    void face::normalize() {
        float norm = 0.0f;
        for (float feature: features) norm += feature * feature;
        norm = std::sqrt(norm);
        for (float &feature: features) feature /= norm;
    }

    facedetector::facedetector() : impl(new impl::face_impl()) {
    }

    facedetector::~facedetector() {
        delete impl;
    }

    //
    // std::ostream &operator<<(std::ostream &lhs, const facedetector &rhs) {
    //     return lhs << rhs.to_json().dump();
    // }

    void facedetector::set_threshold(const double threshold) {
        impl->set_threshold(threshold);
    }

    void facedetector::set_max_faces(const int max_faces) {
        impl->set_max_faces(max_faces);
    }

    void facedetector::detect(const filesystem::path &image_path) const {
        impl->run(image_path.string());
    }

    bool facedetector::display(const int timeout) const {
        return impl->annotated().show(timeout);
    }

    const vector<image> &facedetector::faces() const {
        return impl->faces;
    }

    const vector<face> &facedetector::features() const {
        return impl->features;
    }

    // nlohmann::ordered_json facedetector::to_json() const {
    //     return impl->json();
    // }
} // namespace sc
