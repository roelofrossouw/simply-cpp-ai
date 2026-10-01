#include "ocr.h"

#include "config.h"
#include "image.h"
#include "onnx.h"
#include <dbscan.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace fs = std::filesystem;

namespace sc {
    namespace impl {
        namespace {
            const fs::path model_dir{SIMPLY_CPP_MODEL_DIR};
            constexpr double AngleEpsilon = 10.0;

            fs::path find_dictionary(const std::string &recognition_model, const fs::path &dictionary) {
                if (!dictionary.empty()) {
                    if (!fs::is_regular_file(dictionary))
                        throw std::runtime_error{"OCR dictionary not found: " + dictionary.string()};
                    return dictionary;
                }

                fs::path model_path{recognition_model};
                if (model_path.extension() == ".onnx") model_path.replace_extension();
                const auto filename = model_path.filename().string() + ".txt";
                const auto beside_model = model_path.parent_path() / filename;
                if (fs::is_regular_file(beside_model)) return beside_model;

                const auto installed_model = model_dir / filename;
                if (fs::is_regular_file(installed_model)) return installed_model;

                throw std::runtime_error{"OCR dictionary not found: " + installed_model.string()};
            }

            std::vector<std::string> load_dictionary(const fs::path &path) {
                std::ifstream input{path};
                if (!input) throw std::runtime_error{"Could not open OCR dictionary: " + path.string()};

                std::vector<std::string> characters{""}; // CTC blank class
                std::string character;
                while (std::getline(input, character)) characters.push_back(character);
                if (input.bad()) throw std::runtime_error{"Could not read OCR dictionary: " + path.string()};
                characters.emplace_back(" ");
                return characters;
            }

            double normalized_angle(double angle) {
                angle = std::fmod(angle, 180.0);
                if (angle < 0) angle += 180;
                return angle;
            }

            std::vector<rotated_rect> likely_text_rectangles(const std::vector<rotated_rect> &rectangles,
                                                             const size_t minimum_neighbors,
                                                             const double maximum_height_difference) {
                std::vector<double> angles;
                angles.reserve(rectangles.size());
                for (const auto &rectangle: rectangles) angles.push_back(normalized_angle(rectangle.angle()));
                const auto angle_labels = dbscan(angles, AngleEpsilon, minimum_neighbors, 180);
                const int angle_cluster_count = angle_labels.empty()
                                                    ? 0
                                                    : *std::ranges::max_element(angle_labels) + 1;
                std::vector<bool> keep(rectangles.size());
                for (int angle_cluster = 0; angle_cluster < angle_cluster_count; ++angle_cluster) {
                    std::vector<size_t> members;
                    std::vector<double> heights;
                    for (size_t i = 0; i < angle_labels.size(); ++i) {
                        if (angle_labels[i] != angle_cluster) continue;
                        members.push_back(i);
                        heights.push_back(rectangles[i].height());
                    }
                    if (members.size() < minimum_neighbors) continue;

                    const auto height_labels = dbscan(heights, maximum_height_difference, minimum_neighbors);
                    const int height_cluster_count = height_labels.empty()
                                                         ? 0
                                                         : *std::ranges::max_element(height_labels) + 1;
                    for (int height_cluster = 0; height_cluster < height_cluster_count; ++height_cluster) {
                        for (size_t i = 0; i < height_labels.size(); ++i) {
                            if (height_labels[i] != height_cluster) continue;
                            keep[members[i]] = true;
                        }
                    }
                }

                std::vector<rotated_rect> result;
                result.reserve(rectangles.size());
                for (size_t i = 0; i < rectangles.size(); ++i)
                    if (keep[i]) result.push_back(rectangles[i]);
                return result;
            }

            rotated_rect map_to_original(const rotated_rect &bounds, const image &detector_input,
                                         const image &original, const size_i &map_size) {
                const auto padding = detector_input.padding();
                const auto content = detector_input.cropped_size();
                const double scale_x = static_cast<double>(detector_input.size().width()) / map_size.width() *
                                       original.size().width() / content.width();
                const double scale_y = static_cast<double>(detector_input.size().height()) / map_size.height() *
                                       original.size().height() / content.height();
                const double center_x = (bounds.center().x() * detector_input.size().width() / map_size.width() -
                                         padding.width()) * original.size().width() / content.width();
                const double center_y = (bounds.center().y() * detector_input.size().height() / map_size.height() -
                                         padding.height()) * original.size().height() / content.height();
                const double angle = bounds.angle() * std::numbers::pi / 180.0;
                const double width_scale = std::hypot(std::cos(angle) * scale_x, std::sin(angle) * scale_y);
                const double height_scale = std::hypot(std::sin(angle) * scale_x, std::cos(angle) * scale_y);
                double width = bounds.width() * width_scale;
                double height = bounds.height() * height_scale;
                double mapped_angle = std::atan2(std::sin(angle) * scale_y, std::cos(angle) * scale_x) *
                                      180.0 / std::numbers::pi;
                if (height > width) {
                    std::swap(width, height);
                    mapped_angle += 90;
                }
                return {
                    {center_x, center_y},
                    {width, height},
                    normalized_angle(mapped_angle)
                };
            }

            void validate_minimum_confidence(const double minimum_confidence) {
                if (!std::isfinite(minimum_confidence) || minimum_confidence < 0 || minimum_confidence > 100)
                    throw std::invalid_argument{"OCR minimum confidence must be finite and between 0 and 100"};
            }
        }

        class ocr_detector_impl {
        public:
            explicit ocr_detector_impl(const std::string &model) : detector(model, false) {
#ifndef NDEBUG
                detector.show_shapes();
#endif
            }

            void set_threshold(const double threshold) {
                threshold_ = {threshold, 2};
            }

            void set_minimum_neighbors(const size_t minimum_neighbors) {
                if (minimum_neighbors == 0)
                    throw std::invalid_argument{"OCR minimum neighbor count must be greater than zero"};
                minimum_neighbors_ = minimum_neighbors;
            }

            void set_maximum_height_difference(const double difference) {
                if (!std::isfinite(difference) || difference < 0)
                    throw std::invalid_argument{"OCR maximum height difference must be finite and non-negative"};
                maximum_height_difference_ = difference;
            }

            [[nodiscard]] std::vector<rotated_rect> detect(const image &input) const {
                if (input.empty()) throw std::invalid_argument{"Cannot detect text in an empty image"};
                image detector_input{input};
                detector_input.snap_to_size(32, {detector_input.size()});

                const auto outputs = detector.process_image(detector_input);
                if (outputs.size() != 1 || outputs.front().shape.size() != 4 ||
                    outputs.front().shape[0] != 1 || outputs.front().shape[1] != 1)
                    throw std::runtime_error{"PaddleOCR detection model must return a [1, 1, height, width] map"};

                const auto &output = outputs.front();
                const size_i map_size{static_cast<int>(output.shape[3]), static_cast<int>(output.shape[2])};
                if (map_size.width() <= 0 || map_size.height() <= 0)
                    throw std::runtime_error{"PaddleOCR detection model returned an invalid map size"};

                image probability_map = image::from_blob(output.data, map_size.width(), map_size.height(), 1);
                probability_map.mask(255.0 * static_cast<double>(threshold_) / 100.0);
                auto rectangles = probability_map.find_min_area_rects(500);
                rectangles = likely_text_rectangles(rectangles, minimum_neighbors_, maximum_height_difference_);

                std::vector<rotated_rect> result;
                result.reserve(rectangles.size());
                for (const auto &bounds: rectangles) {
                    const auto mapped = map_to_original(bounds, detector_input, input, map_size);
                    result.emplace_back(
                        mapped.center(),
                        size{mapped.width() + mapped.height() * 2, mapped.height() * 2},
                        mapped.angle());
                }
                std::ranges::sort(result, [](const rotated_rect &lhs, const rotated_rect &rhs) {
                    const auto lhs_box = static_cast<rect>(lhs);
                    const auto rhs_box = static_cast<rect>(rhs);
                    if (lhs_box.top() != rhs_box.top()) return lhs_box.top() < rhs_box.top();
                    return lhs_box.left() < rhs_box.left();
                });
                return result;
            }

        private:
            onnx detector;
            percent threshold_{50, 2};
            size_t minimum_neighbors_{5};
            double maximum_height_difference_{30};
        };

        class ocr_recognizer_impl {
        public:
            ocr_recognizer_impl(const std::string &model, const fs::path &dictionary)
                : recognizer(model, false),
                  characters(load_dictionary(find_dictionary(model, dictionary))) {
#ifndef NDEBUG
                recognizer.show_shapes();
#endif
            }

            void set_threshold(const double threshold) {
                validate_minimum_confidence(threshold);
                threshold_ = {threshold, 2};
            }

            [[nodiscard]] ocr_recognizer::result recognize(const image &input) const {
                if (input.empty()) throw std::invalid_argument{"Cannot recognise an empty image"};
                image resized{input};
                resized.resize_to({0, 48});
                return decode(recognizer.process_image(resized));
            }

            [[nodiscard]] std::vector<ocr_recognizer::result> recognize(const std::vector<image> &inputs) const {
                if (inputs.empty()) return {};
                std::vector<image> resized;
                resized.reserve(inputs.size());
                for (const auto &input: inputs) {
                    if (input.empty()) throw std::invalid_argument{"Cannot recognise an empty image"};
                    auto &text_image = resized.emplace_back(input);
                    text_image.resize_to({0, 48});
                }
                const auto outputs = recognizer.process_images(resized);
                if (outputs.size() != 1 || outputs.front().shape.size() != 3 ||
                    outputs.front().shape[0] != static_cast<int64_t>(resized.size()))
                    throw std::runtime_error{"PaddleOCR recognition model returned an invalid batch"};

                std::vector<ocr_recognizer::result> result;
                result.reserve(resized.size());
                for (size_t i = 0; i < resized.size(); ++i) {
                    auto recognized = decode(outputs, i);
                    if (recognized.confidence >= threshold_) result.push_back(std::move(recognized));
                }
                return result;
            }

        private:
            [[nodiscard]] ocr_recognizer::result decode(const std::vector<output> &outputs,
                                                        const size_t batch_index = 0) const {
                if (outputs.size() != 1 || outputs.front().shape.size() != 3 ||
                    outputs.front().shape[0] <= static_cast<int64_t>(batch_index) ||
                    outputs.front().shape[1] <= 0 || outputs.front().shape[2] <= 0)
                    throw std::runtime_error{
                        "PaddleOCR recognition model must return a [batch, time, classes] tensor"
                    };

                const auto &output = outputs.front();
                const auto timesteps = static_cast<int>(output.shape[1]);
                const auto classes = static_cast<int>(output.shape[2]);
                const auto character_size = characters.size();

                ocr_recognizer::result result;
                double confidence_sum{};
                int recognized_characters{};
                int previous_class{};
                size_t blank_run{};
                std::vector<int> decoded_classes;
                std::vector<size_t> blank_gaps;
                for (int timestep = 0; timestep < timesteps; ++timestep) {
                    const auto *scores = output.data + (batch_index * timesteps + timestep) * classes;
                    int best_class{};
                    float best_score = scores[0];
                    for (int class_index = 1; class_index < character_size; ++class_index) {
                        if (scores[class_index] > best_score) {
                            best_score = scores[class_index];
                            best_class = class_index;
                        }
                    }
                    if (best_class == 0) {
                        ++blank_run;
                    } else {
                        if (best_class != previous_class) {
                            decoded_classes.push_back(best_class);
                            blank_gaps.push_back(blank_run);
                        }
                        blank_run = 0;
                    }
                    if (best_class != 0 && best_class != previous_class) {
                        confidence_sum += best_score;
                        ++recognized_characters;
                    }
                    previous_class = best_class;
                }
                std::vector<size_t> nonzero_gaps;
                for (const size_t gap: blank_gaps)
                    if (gap > 0) nonzero_gaps.push_back(gap);
                size_t typical_gap{};
                if (!nonzero_gaps.empty()) {
                    std::ranges::sort(nonzero_gaps);
                    typical_gap = nonzero_gaps[nonzero_gaps.size() / 2];
                }
                const size_t word_gap = std::max<size_t>(9, typical_gap * 3);
                const bool alphabetic = std::ranges::all_of(decoded_classes, [this](const int class_index) {
                    return std::ranges::all_of(characters[class_index], [](const unsigned char c) {
                        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                    });
                });
                for (size_t i = 0; i < decoded_classes.size(); ++i) {
                    if (alphabetic && i && blank_gaps[i] >= word_gap &&
                        characters[decoded_classes[i]] != " " && result.text.back() != ' ')
                        result.text += ' ';
                    result.text += characters[decoded_classes[i]];
                }
                if (recognized_characters) result.confidence = {confidence_sum / recognized_characters * 100.0, 2};
                return result;
            }

            onnx recognizer;
            std::vector<std::string> characters;
            percent threshold_{0, 2};
        };

        class ocr_impl {
        public:
            ocr_impl(const std::string &detection_model, const std::string &recognition_model,
                     const fs::path &dictionary)
                : detector(detection_model), recognizer(recognition_model, dictionary) {
            }

            void set_threshold(const double threshold) {
                detector.set_threshold(threshold);
            }

            void set_minimum_neighbors(const size_t minimum_neighbors) {
                detector.set_minimum_neighbors(minimum_neighbors);
            }

            void set_maximum_height_difference(const double difference) {
                detector.set_maximum_height_difference(difference);
            }

            void run(const fs::path &image_path, const double minimum_confidence) {
                validate_minimum_confidence(minimum_confidence);
                const image input{image_path.string()};
                run(input, minimum_confidence);
            }

            void run(const image &input, const double minimum_confidence) {
                validate_minimum_confidence(minimum_confidence);
                lines_.clear();
                original = std::make_unique<image>(input);
                const auto regions = detector.detect(*original);
                const auto text_images = detector.text_images(*original, regions);
                std::vector<image> orientation_images;
                orientation_images.reserve(text_images.size() * 2);
                for (const auto &text_image: text_images) {
                    orientation_images.push_back(text_image);
                    auto &upside_down = orientation_images.emplace_back(text_image);
                    upside_down.rotate(180);
                }
                const auto recognized_images = recognizer.recognize(orientation_images);
                for (size_t i = 0; i < regions.size(); ++i) {
                    const auto &forward = recognized_images[i * 2];
                    const auto &upside_down = recognized_images[i * 2 + 1];
                    const auto &recognized = upside_down.confidence > forward.confidence ? upside_down : forward;
                    if (recognized.text.empty() || recognized.confidence < minimum_confidence) continue;
                    lines_.push_back({
                        recognized.text, recognized.confidence, static_cast<rect>(regions[i])
                    });
                }
            }

            [[nodiscard]] nlohmann::ordered_json json() const {
                nlohmann::ordered_json result = nlohmann::ordered_json::array();
                for (const auto &line: lines_) {
                    result.push_back({
                        {"text", line.text},
                        {"confidence", static_cast<double>(line.confidence)},
                        {
                            "box", {
                                {"x", line.box.left()},
                                {"y", line.box.top()},
                                {"width", line.box.width()},
                                {"height", line.box.height()}
                            }
                        }
                    });
                }
                return result;
            }

            [[nodiscard]] std::string text() const {
                std::string result;
                for (const auto &line: lines_) {
                    if (!result.empty()) result += '\n';
                    result += line.text;
                }
                return result;
            }

            [[nodiscard]] image annotated() const {
                if (!original) throw std::runtime_error{"No image has been detected"};
                image result{*original};
                for (const auto &line: lines_) {
                    result.rect(line.box);
                    result.text(line.text, {
                                    static_cast<int>(line.box.left()),
                                    std::max(20, static_cast<int>(line.box.top()) - 5)
                                });
                }
                return result;
            }

            [[nodiscard]] const std::vector<ocr::line> &lines() const noexcept {
                return lines_;
            }

        private:
            ocr_detector detector;
            ocr_recognizer recognizer;
            std::unique_ptr<image> original;
            std::vector<ocr::line> lines_;
        };
    }

    ocr_detector::ocr_detector(const std::string &model) : impl(new impl::ocr_detector_impl(model)) {
    }

    ocr_detector::~ocr_detector() {
        delete impl;
    }

    void ocr_detector::set_threshold(const double threshold) {
        impl->set_threshold(threshold);
    }

    void ocr_detector::set_minimum_neighbors(const size_t minimum_neighbors) {
        impl->set_minimum_neighbors(minimum_neighbors);
    }

    void ocr_detector::set_maximum_height_difference(const double maximum_height_difference) {
        impl->set_maximum_height_difference(maximum_height_difference);
    }

    std::vector<rotated_rect> ocr_detector::detect(const fs::path &image_path) const {
        return detect(image{image_path.string()});
    }

    std::vector<rotated_rect> ocr_detector::detect(const image &input) const {
        return impl->detect(input);
    }

    std::vector<image> ocr_detector::text_images(const image &input,
                                                 const std::vector<rotated_rect> &regions) const {
        if (input.empty()) throw std::invalid_argument{"Cannot extract text from an empty image"};
        std::vector<image> result;
        result.reserve(regions.size());
        for (const auto &region: regions) result.push_back(input.deskewed(region));
        return result;
    }

    std::vector<image> ocr_detector::text_images(const fs::path &image_path) const {
        return text_images(image{image_path.string()});
    }

    std::vector<image> ocr_detector::text_images(const image &input) const {
        return text_images(input, detect(input));
    }

    ocr_recognizer::ocr_recognizer(const std::string &model, const fs::path &dictionary)
        : impl(new impl::ocr_recognizer_impl(model, dictionary)) {
    }

    ocr_recognizer::~ocr_recognizer() {
        delete impl;
    }

    void ocr_recognizer::set_threshold(const double threshold) {
        impl->set_threshold(threshold);
    }

    ocr_recognizer::result ocr_recognizer::recognize(const image &input) const {
        return impl->recognize(input);
    }

    std::vector<ocr_recognizer::result> ocr_recognizer::recognize(const std::vector<image> &inputs) const {
        return impl->recognize(inputs);
    }

    ocr::ocr(const std::string &detection_model, const std::string &recognition_model,
             const std::filesystem::path &dictionary)
        : impl(new impl::ocr_impl(detection_model, recognition_model, dictionary)) {
    }

    ocr::~ocr() {
        delete impl;
    }

    ocr::operator std::string() const {
        return impl->json().dump();
    }

    ocr::operator nlohmann::ordered_json() const {
        return impl->json();
    }

    std::ostream &operator<<(std::ostream &lhs, const ocr &rhs) {
        return lhs << static_cast<std::string>(rhs);
    }

    void ocr::set_threshold(const double threshold) {
        impl->set_threshold(threshold);
    }

    void ocr::set_minimum_neighbors(const size_t minimum_neighbors) {
        impl->set_minimum_neighbors(minimum_neighbors);
    }

    void ocr::set_maximum_height_difference(const double maximum_height_difference) {
        impl->set_maximum_height_difference(maximum_height_difference);
    }

    void ocr::detect(const std::filesystem::path &image_path, const double minimum_confidence) const {
        impl->run(image_path, minimum_confidence);
    }

    void ocr::detect(const image &input, const double minimum_confidence) const {
        impl->run(input, minimum_confidence);
    }

    const std::vector<ocr::line> &ocr::lines() const {
        return impl->lines();
    }

    std::string ocr::text() const {
        return impl->text();
    }

    bool ocr::display(const int timeout) const {
        return impl->annotated().show(timeout);
    }

    nlohmann::ordered_json ocr::to_json() const {
        return impl->json();
    }
} // namespace sc
