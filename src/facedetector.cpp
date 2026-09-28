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

        struct layer_info {
            const int stride;
            const size_i grid_size;

            layer_info(const int stride, const size_i &image_size)
                : stride(stride),
                  grid_size(get_grid_size(stride, image_size)) {
            }

            [[nodiscard]] int detection_count() const {
                return grid_size.area() * 2;
            }

            [[nodiscard]] point cell_center(const int detection_index) const {
                const int cell_index = detection_index / 2;
                return {
                    cell_index % grid_size.width() * stride,
                    cell_index / grid_size.width() * stride
                };
            }

        private:
            static size_i get_grid_size(const int stride, const size_i &image_size) {
                if (stride <= 0 || image_size.width() % stride != 0 || image_size.height() % stride != 0)
                    throw invalid_argument{"Face detector image dimensions must be divisible by the stride"};
                return {image_size.width() / stride, image_size.height() / stride};
            }
        };

        struct detection {
            percent score;
            array<point, 5> landmarks;
            rect box;

            detection(const float confidence, const int index,
                      const layer_info &layer, const float *box_data,
                      const float *landmark_data) : score(confidence, 0) {
                const point center = layer.cell_center(index);
                const float *box_offsets = box_data + index * 4;
                box = rect::ltrb(-box_offsets[0], -box_offsets[1], box_offsets[2], box_offsets[3]) *
                      layer.stride + center;
                for (size_t i = 0; i < landmarks.size(); ++i) {
                    landmarks[i] = center + point{
                                       landmark_data[index * 10 + i * 2],
                                       landmark_data[index * 10 + i * 2 + 1]
                                   } * layer.stride;
                }
            }

            [[nodiscard]] bool overlaps(const detection &rhs) const {
                return box.iou(rhs.box) > 0.2;
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

            void set_threshold(const float threshold) { threshold_ = threshold; }

            void set_max_faces(const int max_faces) { max_faces_ = max_faces; }

            static image extract_face(const detection &detection, const image &img) {
                return img.warp(detection.landmarks, ARC_FACE_TEMPLATE, {112, 112});
            }

            static void annotate_face(const detection &detection, image &img) {
                img.rect(detection.box);
                const point_i label_position{detection.box.left(), std::max(20, (int) detection.box.top() - 5)};
                const std::string label = detection.score;
                img.text(label, label_position);
                for (const auto &landmark: detection.landmarks) img.circle({landmark.x(), landmark.y()}, 1);
            }

            vector<face> process_detections(const vector<const float *> &outputs, const image &img,
                                            const bool include_face_images) {
                vector<layer_info> layers;
                layers.reserve(STRIDES.size());
                const auto image_size = img.size();
                for (const int stride_size: STRIDES) layers.emplace_back(stride_size, image_size);

                vector<detection> candidates;
                for (size_t layer = 0; layer < layers.size(); ++layer) {
                    const float *scores = outputs[layer];
                    for (int index = 0; index < layers[layer].detection_count(); ++index) {
                        if (scores[index] < threshold_) continue;
                        candidates.emplace_back(scores[index], index, layers[layer],
                                                outputs[3 + layer],
                                                outputs[6 + layer]);
                    }
                }

                ranges::sort(candidates, [](const detection &lhs, const detection &rhs) {
                    return lhs.score > rhs.score;
                });

                for (const auto &candidate: candidates) {
                    const auto overlaps = ranges::any_of(detections, [&candidate](const detection &selected_detection) {
                        return candidate.overlaps(selected_detection);
                    });
                    if (overlaps) continue;
                    detections.push_back(candidate);
                    if (detections.size() == max_faces_) break;
                }
                vector<face> result;
                result.reserve(detections.size());
                for (const auto &detection: detections) {
                    auto aligned_face = extract_face(detection, img);
                    const auto features = extractor.process_image(aligned_face)[0].data;
                    if (include_face_images) result.emplace_back(features, aligned_face);
                    else result.emplace_back(features);
                }
                return result;
            }

            vector<face> run(const string &image_filename, const bool include_face_images) {
                detections.clear();
                original = make_unique<image>(image_filename);
                image input{*original};
                input.snap_to_size(stride, valid_sizes);
                auto out = detector.process_image(input);
                vector<const float *> result;
                for (const auto &o: out) result.push_back(o.data);
                return process_detections(result, input, include_face_images);
            }

            [[nodiscard]] image annotated() const {
                if (!original) throw runtime_error("No image has been detected");
                image result{*original};
                result.snap_to_size(stride, valid_sizes);
                for (const auto &detection: detections) annotate_face(detection, result);
                return result;
            }

            onnx detector; // Model to locate the face
            onnx extractor; // Model to extract features
            vector<detection> detections;
            unique_ptr<image> original;

        private:
            float threshold_{0.6f};
            size_t max_faces_{2};
        };
    }

    facedetector::facedetector() : impl(new impl::face_impl()) {
    }

    facedetector::~facedetector() {
        delete impl;
    }

    void facedetector::set_threshold(const double threshold) {
        impl->set_threshold(threshold);
    }

    void facedetector::set_max_faces(const int max_faces) {
        impl->set_max_faces(max_faces);
    }

    vector<face> facedetector::detect(const filesystem::path &image_path, const bool include_face_images) const {
        return impl->run(image_path.string(), include_face_images);
    }

    bool facedetector::display(const int timeout) const {
        return impl->annotated().show(timeout);
    }
} // namespace sc
