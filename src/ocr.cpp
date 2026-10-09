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
            // The detector reads text best when it is about TypicalTextHeight pixels tall in its
            // text map (which marks a narrow core of each line). Much larger, it sees the gaps between
            // words as wide as columns and returns words rather than lines, so text taller than
            // MaximumTextHeight is looked at again, scaled down. Inputs are first limited to
            // MaximumDetectionSide pixels on the longer side - PaddleOCR limits its own the same way
            // (det_limit_side_len) - which also keeps detection on large photos quick.
            constexpr int MaximumDetectionSide = 1280;
            constexpr double MaximumTextHeight = 14;
            constexpr double TypicalTextHeight = 11;
            constexpr double MinimumTextHeight = 6;
            // Tiles of a large input overlap this much (a multiple of 32), more than a line is tall.
            constexpr int TileOverlap = 256;
            // How PaddleOCR's models were trained to see pixels, as (pixel - mean) * scale: the
            // detector with ImageNet's normalisation (mean 0.45, standard deviation 0.226, as one
            // value for all three channels), the recogniser from -1 to 1. The default (-0.5 to
            // 0.5) gave the detector about a quarter of the contrast it expects.
            constexpr double DetectionScale = 1.0 / (0.226 * 255);
            constexpr double DetectionMean = 0.45 * 255;
            constexpr double RecognitionScale = 1.0 / 127.5;
            constexpr double RecognitionMean = 127.5;
            // PaddleOCR's text line orientation classifier, and how sure it must be for a line to be
            // read only that way up (otherwise it is read both ways, which takes twice as long).
            const std::string LineOrientationModel = "paddle_line_rotate.onnx";
            constexpr float LineOrientationSure = 0.9f;
            // A line read one way only is read the other way as well when it reads below this.
            constexpr double RereadBelow = 98;
            // A region this many times longer than tall is clearly a line, and shows the page's text
            // direction; one less than ShortRegionAspect long is short enough to be turned to it.
            constexpr double LongRegionAspect = 3;
            // How far a region is grown back out from the text map's shrunk core (PaddleOCR uses
            // 1.5; 2 covered whole lines best in testing, line ends included).
            constexpr double UnclipRatio = 2;
            constexpr double ShortRegionAspect = 2;
            // Smaller regions in the text map are dropped as specks. Small enough to keep a lone
            // letter - the M or F under an identity card's Sex label - which 500 lost.
            constexpr double MinimumRegionArea = 50;

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

            // Drops regions whose angle or height doesn't match any group of others: specks and
            // lines on a busy document. A page with only a few regions gives DBSCAN too little to
            // go on - with fewer regions than minimum_neighbors every one would be noise - so the
            // neighbour count is capped at the number of regions, and when no group forms at all,
            // nothing is dropped.
            std::vector<rotated_rect> likely_text_rectangles(const std::vector<rotated_rect> &rectangles,
                                                             const size_t minimum_neighbors,
                                                             const double maximum_height_difference) {
                if (rectangles.empty()) return {};
                const size_t neighbors = std::min(minimum_neighbors, rectangles.size());
                std::vector<double> angles;
                angles.reserve(rectangles.size());
                for (const auto &rectangle: rectangles) angles.push_back(normalized_angle(rectangle.angle()));
                const auto angle_labels = dbscan(angles, AngleEpsilon, neighbors, 180);
                const int angle_cluster_count = *std::ranges::max_element(angle_labels) + 1;
                if (angle_cluster_count == 0) return rectangles;

                std::vector<bool> keep(rectangles.size());
                for (int angle_cluster = 0; angle_cluster < angle_cluster_count; ++angle_cluster) {
                    std::vector<size_t> members;
                    std::vector<double> heights;
                    for (size_t i = 0; i < angle_labels.size(); ++i) {
                        if (angle_labels[i] != angle_cluster) continue;
                        members.push_back(i);
                        heights.push_back(rectangles[i].height());
                    }

                    const auto height_labels = dbscan(heights, maximum_height_difference,
                                                      std::min(neighbors, members.size()));
                    const bool any_height_cluster = *std::ranges::max_element(height_labels) >= 0;
                    for (size_t i = 0; i < height_labels.size(); ++i)
                        if (height_labels[i] >= 0 || !any_height_cluster) keep[members[i]] = true;
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
                return {
                    {center_x, center_y},
                    {width, height},
                    normalized_angle(mapped_angle)
                };
            }

            // A region's direction is taken from its longer side, but a lone letter or a short word
            // can be as tall as it is wide or taller (F, 7), so its longer side may run across the
            // text: it was then dropped as noise among the page's lines, or read sideways. Short
            // regions are turned to run the way the page's text does: the median direction of its
            // clearly long regions (or level, without any). Their width can then be the shorter side.
            std::vector<rotated_rect> along_text_direction(std::vector<rotated_rect> regions) {
                std::vector<double> long_directions;
                for (const auto &region: regions)
                    if (region.width() >= region.height() * LongRegionAspect) long_directions.push_back(normalized_angle(region.angle()));
                double direction = 0;
                if (!long_directions.empty()) {
                    std::ranges::nth_element(long_directions, long_directions.begin() + long_directions.size() / 2);
                    direction = long_directions[long_directions.size() / 2];
                }
                for (auto &region: regions) {
                    const double longer = std::max(region.width(), region.height());
                    const double shorter = std::max(1e-9, std::min(region.width(), region.height()));
                    if (longer >= shorter * ShortRegionAspect) continue;
                    const double across = std::fmod(std::abs(normalized_angle(region.angle()) - direction), 180.0);
                    if (std::min(across, 180 - across) <= 45) continue;
                    region = {region.center(), size{region.height(), region.width()}, normalized_angle(region.angle() + 90)};
                }
                return regions;
            }

            // Top to bottom, then left to right within a row. Sorting on the top edge alone puts
            // a word a pixel higher first, so regions whose centres lie within half a height of a
            // row's first region share that row and are ordered by their left edge.
            std::vector<rotated_rect> in_reading_order(std::vector<rotated_rect> regions) {
                std::ranges::sort(regions, {}, [](const rotated_rect &region) { return region.center().y(); });
                std::vector<rotated_rect> ordered;
                ordered.reserve(regions.size());
                for (auto row_start = regions.begin(); row_start != regions.end();) {
                    const double row_y = row_start->center().y();
                    const double tolerance = static_cast<rect>(*row_start).height() / 2;
                    auto row_end = std::find_if(row_start, regions.end(), [&](const rotated_rect &region) {
                        return region.center().y() - row_y > tolerance;
                    });
                    std::sort(row_start, row_end, [](const rotated_rect &lhs, const rotated_rect &rhs) {
                        return static_cast<rect>(lhs).left() < static_cast<rect>(rhs).left();
                    });
                    ordered.insert(ordered.end(), row_start, row_end);
                    row_start = row_end;
                }
                return ordered;
            }

            void validate_minimum_confidence(const double minimum_confidence) {
                if (!std::isfinite(minimum_confidence) || minimum_confidence < 0 || minimum_confidence > 100)
                    throw std::invalid_argument{"OCR minimum confidence must be finite and between 0 and 100"};
            }
        }

        class ocr_detector_impl {
        public:
            explicit ocr_detector_impl(const std::string &model) : detector(model, true) { // CoreML on macOS: about 3x quicker
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
                const int longest_side = std::max(input.size().width(), input.size().height());
                const double scale = std::min(1.0, static_cast<double>(MaximumDetectionSide) / longest_side);
                auto found = find_regions(input, scale);
                // Large text is found word by word, and text scaled down too far is lost: look again
                // at the scale the model reads best - for small text in a large image, larger than the
                // first look, in tiles.
                const double height = median_height(found.regions);
                if (height > MaximumTextHeight || (height > 0 && height < MinimumTextHeight && scale < 1)) {
                    const double better = std::min(1.0, scale * TypicalTextHeight / height);
                    auto rescaled = find_regions(input, better);
                    if (!rescaled.regions.empty()) found = std::move(rescaled);
                }
                const auto rectangles = likely_text_rectangles(along_text_direction(std::move(found.regions)),
                                                               minimum_neighbors_, maximum_height_difference_);

                std::vector<rotated_rect> result;
                result.reserve(rectangles.size());
                for (const auto &bounds: rectangles) {
                    // The map marks a shrunk core of each line: grow it back by its area times
                    // UnclipRatio over its perimeter on every side, as PaddleOCR's post-processing does.
                    const double w = bounds.width(), h = bounds.height();
                    const double margin = w * h * UnclipRatio / (2 * (w + h));
                    const rotated_rect grown{bounds.center(), size{w + 2 * margin, h + 2 * margin}, bounds.angle()};
                    result.push_back(map_to_original(grown, found.detector_input, input, found.map_size));
                }
                return in_reading_order(std::move(result));
            }

        private:
            // What the model found in the input scaled by scale: text regions in the coordinates of
            // its text map, with what's needed to map them back to the input.
            struct detection {
                image detector_input;
                size_i map_size;
                std::vector<rotated_rect> regions;
            };

            [[nodiscard]] detection find_regions(const image &input, const double scale) const {
                detection found{input, {}, {}};
                if (scale != 1) {
                    found.detector_input.resize_to({
                        std::max(1, static_cast<int>(std::lround(input.size().width() * scale))),
                        std::max(1, static_cast<int>(std::lround(input.size().height() * scale)))
                    });
                }
                found.detector_input.snap_to_size(32, {found.detector_input.size()});
                found.map_size = found.detector_input.size();

                const auto map = text_map(found.detector_input);
                image probability_map = image::from_blob(map.data(), found.map_size.width(), found.map_size.height(), 1);
                probability_map.mask(255.0 * static_cast<double>(threshold_) / 100.0);
                found.regions = probability_map.find_min_area_rects(MinimumRegionArea);
                return found;
            }

            // The model's text probability for every pixel of input (whose sides are multiples of
            // 32). An input larger than MaximumDetectionSide is read in overlapping tiles of that
            // size, whose maps are combined (the higher probability where they overlap), so a line
            // across a tile edge is still one region.
            [[nodiscard]] std::vector<float> text_map(const image &input) const {
                const int width = input.size().width(), height = input.size().height();
                std::vector<float> map(static_cast<size_t>(width) * height, 0.0f);
                const auto starts = [](const int length) {
                    std::vector<int> result{0};
                    if (length <= MaximumDetectionSide) return result;
                    const int stride = MaximumDetectionSide - TileOverlap;
                    for (int start = stride; start + MaximumDetectionSide < length; start += stride) result.push_back(start);
                    result.push_back(length - MaximumDetectionSide); // the last tile ends at the edge
                    return result;
                };
                for (const int top: starts(height)) {
                    for (const int left: starts(width)) {
                        const int tile_width = std::min(width, MaximumDetectionSide);
                        const int tile_height = std::min(height, MaximumDetectionSide);
                        const bool whole = tile_width == width && tile_height == height;
                        const auto outputs = whole
                                                 ? detector.process_image(input, DetectionScale, DetectionMean)
                                                 : detector.process_image(input.cropped(rect_i{left, top, tile_width, tile_height}),
                                                                          DetectionScale, DetectionMean);
                        if (outputs.size() != 1 || outputs.front().shape.size() != 4 ||
                            outputs.front().shape[0] != 1 || outputs.front().shape[1] != 1 ||
                            outputs.front().shape[2] != tile_height || outputs.front().shape[3] != tile_width)
                            throw std::runtime_error{"PaddleOCR detection model must return a [1, 1, height, width] map"};
                        const float *tile = outputs.front().data;
                        for (int y = 0; y < tile_height; ++y) {
                            float *row = map.data() + static_cast<size_t>(top + y) * width + left;
                            const float *tile_row = tile + static_cast<size_t>(y) * tile_width;
                            for (int x = 0; x < tile_width; ++x) row[x] = std::max(row[x], tile_row[x]);
                        }
                    }
                }
                return map;
            }

            static double median_height(const std::vector<rotated_rect> &regions) {
                if (regions.empty()) return 0;
                std::vector<double> heights;
                heights.reserve(regions.size());
                for (const auto &region: regions) heights.push_back(region.height());
                std::ranges::nth_element(heights, heights.begin() + heights.size() / 2);
                return heights[heights.size() / 2];
            }

            onnx detector;
            percent threshold_{50, 2};
            size_t minimum_neighbors_{5};
            double maximum_height_difference_{30};
        };

        class ocr_recognizer_impl {
        public:
            ocr_recognizer_impl(const std::string &model, const fs::path &dictionary)
                : recognizer(model, false), // not CoreML: it compiles again for every new line width
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
                return decode(recognizer.process_image(resized, RecognitionScale, RecognitionMean));
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
                const auto outputs = recognizer.process_images(resized, RecognitionScale, RecognitionMean);
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
                try {
                    line_classifier = std::make_unique<onnx>(LineOrientationModel, false);
                } catch (const std::exception &) {
                    // Not installed (an older simply-cpp-models): every line is read both ways.
                }
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
                // Each line is read the way up the line classifier says, or both ways (keeping the
                // more confident reading) when it isn't sure.
                const auto ways = orientations(text_images);
                std::vector<image> readings;
                readings.reserve(text_images.size() * 2);
                constexpr size_t not_read = static_cast<size_t>(-1);
                std::vector<std::pair<size_t, size_t> > reading_of(text_images.size(), {not_read, not_read});
                for (size_t i = 0; i < text_images.size(); ++i) {
                    if (ways[i] != line_orientation::upside_down) {
                        reading_of[i].first = readings.size();
                        readings.push_back(text_images[i]);
                    }
                    if (ways[i] != line_orientation::upright) {
                        reading_of[i].second = readings.size();
                        readings.emplace_back(text_images[i]).rotate(180);
                    }
                }
                auto recognized_readings = recognizer.recognize(readings);

                // A line read one way only that reads poorly may have been turned the other way:
                // the classifier is wrong on some short lines. Read those the other way too.
                std::vector<image> rereadings;
                std::vector<size_t> reread;
                for (size_t i = 0; i < text_images.size(); ++i) {
                    auto &[forward, upside_down] = reading_of[i];
                    if (forward != not_read && upside_down != not_read) continue;
                    const size_t read = forward != not_read ? forward : upside_down;
                    if (recognized_readings[read].confidence >= RereadBelow) continue;
                    reread.push_back(i);
                    rereadings.emplace_back(text_images[i]);
                    if (forward != not_read) rereadings.back().rotate(180);
                }
                if (!rereadings.empty()) {
                    auto other_way = recognizer.recognize(rereadings);
                    for (size_t r = 0; r < reread.size(); ++r) {
                        auto &[forward, upside_down] = reading_of[reread[r]];
                        (forward == not_read ? forward : upside_down) = recognized_readings.size();
                        recognized_readings.push_back(std::move(other_way[r]));
                    }
                }

                for (size_t i = 0; i < regions.size(); ++i) {
                    const auto [forward, upside_down] = reading_of[i];
                    const auto &recognized = forward == not_read
                                                 ? recognized_readings[upside_down]
                                                 : upside_down == not_read ||
                                                   recognized_readings[forward].confidence >= recognized_readings[upside_down].confidence
                                                       ? recognized_readings[forward]
                                                       : recognized_readings[upside_down];
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
            enum class line_orientation { upright, upside_down, unsure };

            // How each line image is turned, from PaddleOCR's text line orientation classifier:
            // 160 x 80 pixels in, ImageNet-normalised; out, how likely upright and upside down.
            [[nodiscard]] std::vector<line_orientation> orientations(const std::vector<image> &lines) const {
                std::vector<line_orientation> result(lines.size(), line_orientation::unsure);
                if (!line_classifier || lines.empty()) return result;
                std::vector<image> inputs;
                inputs.reserve(lines.size());
                for (const auto &line: lines) inputs.push_back(line.resized({160, 80}));
                const auto outputs = line_classifier->process_images(inputs, DetectionScale, DetectionMean);
                if (outputs.size() != 1 || outputs.front().shape.size() != 2 ||
                    outputs.front().shape[0] != static_cast<int64_t>(lines.size()) || outputs.front().shape[1] != 2)
                    return result;
                const float *scores = outputs.front().data;
                for (size_t i = 0; i < lines.size(); ++i) {
                    if (scores[i * 2] >= LineOrientationSure) result[i] = line_orientation::upright;
                    else if (scores[i * 2 + 1] >= LineOrientationSure) result[i] = line_orientation::upside_down;
                }
                return result;
            }

            ocr_detector detector;
            std::unique_ptr<onnx> line_classifier; // null: read every line both ways
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
