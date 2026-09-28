#include "ocr.h"

#include "config.h"
#include "image.h"
#include "onnx.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

namespace sc {
    namespace impl {
        namespace {
            const fs::path model_dir{SIMPLY_CPP_MODEL_DIR};

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

            int clipped(const int value, const int lower, const int upper) {
                return std::clamp(value, lower, upper);
            }

            rect map_box_to_original(const cv::Rect &box, const image &detector_input, const image &original,
                                     const cv::Size &map_size) {
                const auto padding = detector_input.padding();
                const auto content = detector_input.cropped_size();
                const double x_scale = static_cast<double>(content.width()) / original.size().width();
                const double y_scale = static_cast<double>(content.height()) / original.size().height();
                const double map_x_scale = static_cast<double>(detector_input.size().width()) / map_size.width;
                const double map_y_scale = static_cast<double>(detector_input.size().height()) / map_size.height;

                const double left = (box.x * map_x_scale - padding.x()) / x_scale;
                const double top = (box.y * map_y_scale - padding.y()) / y_scale;
                const double right = ((box.x + box.width) * map_x_scale - padding.x()) / x_scale;
                const double bottom = ((box.y + box.height) * map_y_scale - padding.y()) / y_scale;
                const double x0 = std::clamp(left, 0.0, static_cast<double>(original.size().width()));
                const double y0 = std::clamp(top, 0.0, static_cast<double>(original.size().height()));
                const double x1 = std::clamp(right, x0, static_cast<double>(original.size().width()));
                const double y1 = std::clamp(bottom, y0, static_cast<double>(original.size().height()));
                return rect::ltrb(x0, y0, x1, y1);
            }
        }

        class ocr_impl {
        public:
            ocr_impl(const std::string &detection_model, const std::string &recognition_model,
                     const fs::path &dictionary)
                : detector(detection_model),
                  recognizer(recognition_model),
                  characters(load_dictionary(find_dictionary(recognition_model, dictionary))) {
            }

            void set_threshold(const double threshold) {
                threshold_ = {threshold, 2};
            }

            void run(const fs::path &image_path) {
                const image input{image_path.string()};
                run(input);
            }

            void run(const image &input) {
                lines_.clear();
                original = std::make_unique<image>(input);
                image detector_input{*original};
                detector_input.snap_to_size(64, {detector_input.size()});

                const auto outputs = detector.process_image(detector_input);
                if (outputs.size() != 1 || outputs.front().shape.size() != 4 ||
                    outputs.front().shape[0] != 1 || outputs.front().shape[1] != 1)
                    throw std::runtime_error{"PaddleOCR detection model must return a [1, 1, height, width] map"};

                const auto &output = outputs.front();
                const cv::Size map_size{static_cast<int>(output.shape[3]), static_cast<int>(output.shape[2])};
                if (map_size.width <= 0 || map_size.height <= 0)
                    throw std::runtime_error{"PaddleOCR detection model returned an invalid map size"};

                cv::Mat probability_map(map_size.height, map_size.width, CV_32F, output.data);
                cv::Mat mask;
                cv::threshold(probability_map, mask, static_cast<double>(threshold_) / 100.0, 255,
                              cv::THRESH_BINARY);
                mask.convertTo(mask, CV_8U);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                std::vector<cv::Rect> boxes;
                const auto padding = detector_input.padding();
                const auto content = detector_input.cropped_size();
                const double scale_x = static_cast<double>(detector_input.size().width()) / map_size.width;
                const double scale_y = static_cast<double>(detector_input.size().height()) / map_size.height;
                for (const auto &contour: contours) {
                    if (std::abs(cv::contourArea(contour)) < 100) continue;
                    const auto bounds = cv::boundingRect(contour);
                    const int left = clipped(static_cast<int>(std::floor(bounds.x * scale_x)) - 5,
                                             padding.x(), padding.x() + content.width());
                    const int top = clipped(static_cast<int>(std::floor(bounds.y * scale_y)) - 5,
                                            padding.y(), padding.y() + content.height());
                    const int right = clipped(static_cast<int>(std::ceil((bounds.x + bounds.width) * scale_x)) + 5,
                                              padding.x(), padding.x() + content.width());
                    const int bottom = clipped(static_cast<int>(std::ceil((bounds.y + bounds.height) * scale_y)) + 5,
                                               padding.y(), padding.y() + content.height());
                    if (right > left && bottom > top) boxes.emplace_back(left, top, right - left, bottom - top);
                }

                std::sort(boxes.begin(), boxes.end(), [](const cv::Rect &lhs, const cv::Rect &rhs) {
                    if (lhs.y != rhs.y) return lhs.y < rhs.y;
                    return lhs.x < rhs.x;
                });

                for (const auto &box: boxes) {
                    auto word_image = detector_input.crop({box.x, box.y, box.width, box.height});
                    if (word_image.size().width() < word_image.size().height()) word_image.rotate(90);
                    word_image.resize_to({0, 48});
                    const auto recognized = decode(recognizer.process_image(word_image));
                    if (recognized.text.empty()) continue;
                    lines_.push_back({recognized.text, {recognized.confidence * 100.0, 2},
                                      map_box_to_original(box, detector_input, *original, map_size)});
                }
            }

            [[nodiscard]] nlohmann::ordered_json json() const {
                nlohmann::ordered_json result = nlohmann::ordered_json::array();
                for (const auto &line: lines_) {
                    result.push_back({
                        {"text", line.text},
                        {"confidence", static_cast<double>(line.confidence)},
                        {"box", {
                            {"x", line.box.left()},
                            {"y", line.box.top()},
                            {"width", line.box.width()},
                            {"height", line.box.height()}
                        }}
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
                    result.text(line.text, {static_cast<int>(line.box.left()),
                                            std::max(20, static_cast<int>(line.box.top()) - 5)});
                }
                return result;
            }

            [[nodiscard]] const std::vector<ocr::line> &lines() const noexcept {
                return lines_;
            }

        private:
            struct decoded_text {
                std::string text;
                double confidence{};
            };

            [[nodiscard]] decoded_text decode(const std::vector<output> &outputs) const {
                if (outputs.size() != 1 || outputs.front().shape.size() != 3 ||
                    outputs.front().shape[0] != 1 || outputs.front().shape[1] <= 0 ||
                    outputs.front().shape[2] <= 0)
                    throw std::runtime_error{"PaddleOCR recognition model must return a [1, time, classes] tensor"};

                const auto &output = outputs.front();
                const auto timesteps = static_cast<int>(output.shape[1]);
                const auto classes = static_cast<int>(output.shape[2]);
                if (static_cast<size_t>(classes) > characters.size())
                    throw std::runtime_error{
                        "OCR dictionary has " + std::to_string(characters.size()) +
                        " entries but the recognition model has " + std::to_string(classes) + " classes"};

                decoded_text result;
                double confidence_sum{};
                int recognized_characters{};
                int previous_class{};
                for (int timestep = 0; timestep < timesteps; ++timestep) {
                    const auto *scores = output.data + timestep * classes;
                    int best_class{};
                    float best_score = scores[0];
                    for (int class_index = 1; class_index < classes; ++class_index) {
                        if (scores[class_index] > best_score) {
                            best_score = scores[class_index];
                            best_class = class_index;
                        }
                    }
                    if (best_class != 0 && best_class != previous_class) {
                        result.text += characters[best_class];
                        confidence_sum += best_score;
                        ++recognized_characters;
                    }
                    previous_class = best_class;
                }
                if (recognized_characters) result.confidence = confidence_sum / recognized_characters;
                return result;
            }

            onnx detector;
            onnx recognizer;
            std::vector<std::string> characters;
            std::unique_ptr<image> original;
            std::vector<ocr::line> lines_;
            percent threshold_{50, 2};
        };
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

    void ocr::detect(const std::filesystem::path &image_path) const {
        impl->run(image_path);
    }

    void ocr::detect(const image &input) const {
        impl->run(input);
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
