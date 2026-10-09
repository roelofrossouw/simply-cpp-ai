#include "ocr.h"

#include "config.h"
#include "image.h"
#include "onnx.h"
#include <dbscan.h>

#include <algorithm>
#include <array>
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
            // A line is read the other way as well when it reads below this, and that reading is
            // taken when it is at least OtherWayMargin points more confident. (A run of MRZ fillers read
            // upside down as > could come out 5 to 10 points ahead; a line really upside down reads
            // far worse the wrong way.)
            constexpr double RereadBelow = 98;
            constexpr double OtherWayMargin = 15;
            // How much an MRZ-like line is stretched for its second reading.
            constexpr double DenseLineStretch = 1.5;
            // A region this many times longer than tall is clearly a line, and shows the page's text
            // direction; one less than ShortRegionAspect long is short enough to be turned to it.
            constexpr double LongRegionAspect = 3;
            // How far a region is grown back out from the text map's shrunk core (PaddleOCR uses
            // 1.5; 2 covered whole lines best in testing, line ends included).
            constexpr double UnclipRatio = 2;
            constexpr double ShortRegionAspect = 2;
            // Pieces of one line closer than this many of their heights in the text map are joined.
            constexpr double JoinGapHeights = 1.5;
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
            // The page's text direction, 0 to 180 degrees: the most common direction of its clearly
            // long regions (or level, without any) - a few sideways lines along an edge don't pull
            // it over. Directions repeat every 180 degrees, so 179 and 1 count as neighbours.
            double text_direction(const std::vector<rotated_rect> &regions) {
                constexpr int bins = 36; // 5 degrees each
                std::array<double, bins> weight{};
                for (const auto &region: regions) {
                    if (region.width() < region.height() * LongRegionAspect) continue;
                    weight[static_cast<int>(normalized_angle(region.angle()) / 5) % bins] += region.width();
                }
                const auto peak = static_cast<int>(std::ranges::max_element(weight) - weight.begin());
                if (weight[peak] == 0) return 0;
                // The weighted mean of the directions near the peak, as unit vectors of the doubled angle.
                double x = 0, y = 0;
                const double centre = peak * 5 + 2.5;
                for (const auto &region: regions) {
                    if (region.width() < region.height() * LongRegionAspect) continue;
                    const double away = std::fmod(std::abs(normalized_angle(region.angle()) - centre), 180.0);
                    if (std::min(away, 180 - away) > 15) continue;
                    const double doubled = 2 * region.angle() * std::numbers::pi / 180.0;
                    x += region.width() * std::cos(doubled);
                    y += region.width() * std::sin(doubled);
                }
                return normalized_angle(std::atan2(y, x) / 2 * 180.0 / std::numbers::pi);
            }

            std::vector<rotated_rect> along_text_direction(std::vector<rotated_rect> regions) {
                const double direction = text_direction(regions);
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

            // Pieces of one line that the map left apart - an MRZ line's run of fillers came out as a
            // region of its own - are joined: regions running the same way, about as tall, in line
            // with each other and less than JoinGapHeights of their height apart. Separate columns
            // are much further apart than that.
            std::vector<rotated_rect> joined_line_pieces(std::vector<rotated_rect> regions) {
                const auto same_line = [](const rotated_rect &a, const rotated_rect &b, double &u, double &v) {
                    const double across = std::fmod(std::abs(normalized_angle(a.angle()) - normalized_angle(b.angle())), 180.0);
                    if (std::min(across, 180 - across) > 5) return false;
                    const double height = std::max(a.height(), b.height());
                    if (std::min(a.height(), b.height()) < height * 0.6) return false;
                    const double radians = a.angle() * std::numbers::pi / 180.0;
                    const double dx = b.center().x() - a.center().x(), dy = b.center().y() - a.center().y();
                    u = dx * std::cos(radians) + dy * std::sin(radians);
                    v = -dx * std::sin(radians) + dy * std::cos(radians);
                    if (std::abs(v) > height * 0.3) return false;
                    return std::abs(u) - (a.width() + b.width()) / 2 <= height * JoinGapHeights;
                };
                for (bool joined = true; joined;) {
                    joined = false;
                    for (size_t i = 0; i < regions.size() && !joined; ++i) {
                        for (size_t j = i + 1; j < regions.size() && !joined; ++j) {
                            double u = 0, v = 0;
                            const auto &a = regions[i], &b = regions[j];
                            if (!same_line(a, b, u, v)) continue;
                            // Both in a's frame: a spans -w/2..w/2 along it, b centres at (u, v).
                            const double along_from = std::min(-a.width() / 2, u - b.width() / 2);
                            const double along_to = std::max(a.width() / 2, u + b.width() / 2);
                            const double across_from = std::min(-a.height() / 2, v - b.height() / 2);
                            const double across_to = std::max(a.height() / 2, v + b.height() / 2);
                            const double mid_u = (along_from + along_to) / 2, mid_v = (across_from + across_to) / 2;
                            const double radians = a.angle() * std::numbers::pi / 180.0;
                            regions[i] = {
                                point{a.center().x() + mid_u * std::cos(radians) - mid_v * std::sin(radians),
                                      a.center().y() + mid_u * std::sin(radians) + mid_v * std::cos(radians)},
                                size{along_to - along_from, across_to - across_from}, a.angle()
                            };
                            regions.erase(regions.begin() + static_cast<std::ptrdiff_t>(j));
                            joined = true;
                        }
                    }
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

        // The model puts a space before some hyphens and apostrophes inside a word or number
        // (1994 -04, ANNA -MARIE, O 'NEILL): drop a space between a letter or digit and a hyphen
        // or apostrophe joined to what follows. One with spaces on both sides (a - b) stays.
        static std::string without_stray_spaces(const std::string &text) {
            const auto alnum = [](const char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; };
            std::string result;
            result.reserve(text.size());
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] == ' ' && i > 0 && i + 2 < text.size() && alnum(text[i - 1]) &&
                    (text[i + 1] == '-' || text[i + 1] == '\'') && alnum(text[i + 2]))
                    continue;
                result += text[i];
            }
            return result;
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
                const auto rectangles = likely_text_rectangles(joined_line_pieces(along_text_direction(std::move(found.regions))),
                                                               minimum_neighbors_, maximum_height_difference_);

                std::vector<rotated_rect> result;
                result.reserve(rectangles.size());
                for (const auto &bounds: rectangles) {
                    // The map marks a shrunk core of each line: grow it back by its area times
                    // UnclipRatio over its perimeter on every side, as PaddleOCR's post-processing does.
                    const double w = bounds.width(), h = bounds.height();
                    const double margin = w * h * UnclipRatio / (2 * (w + h));
                    const rotated_rect grown{bounds.center(), size{w + 2 * margin, h + 2 * margin}, bounds.angle()};
                    auto mapped = map_to_original(grown, found.detector_input, input, found.map_size);
                    // Within 90 degrees of level, so an upright page's lines are all cut out the same
                    // way up: kept between 0 and 180 degrees, a line tilted a little anticlockwise,
                    // at 179, was cut out upside down. (sc::ocr turns the image upright first.)
                    double angle = mapped.angle();
                    while (angle > 90) angle -= 180;
                    while (angle <= -90) angle += 180;
                    mapped.angle(angle);
                    result.push_back(mapped);
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

            [[nodiscard]] std::vector<ocr_recognizer::result> recognize(const std::vector<image> &inputs,
                                                                        const double stretch = 1) const {
                if (inputs.empty()) return {};
                std::vector<image> resized;
                resized.reserve(inputs.size());
                for (const auto &input: inputs) {
                    if (input.empty()) throw std::invalid_argument{"Cannot recognise an empty image"};
                    auto &text_image = resized.emplace_back(input);
                    text_image.resize_to({0, 48});
                    if (stretch != 1) text_image.resize_to({static_cast<int>(std::lround(text_image.size().width() * stretch)), 48});
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
                result.text = without_stray_spaces(result.text);
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

            void set_auto_rotate(const bool value) { auto_rotate = value; }

            [[nodiscard]] int rotation() const noexcept { return rotation_; }

            void run(const image &input, const double minimum_confidence) {
                validate_minimum_confidence(minimum_confidence);
                lines_.clear();
                original = std::make_unique<image>(input);
                // Read upright: then all of a page's lines are cut out the same way up.
                rotation_ = auto_rotate ? ocr::upright_rotation(input) : 0;
                image upright{input};
                if (rotation_) upright.rotate(rotation_);
                const auto regions = detector.detect(upright);
                const auto text_images = detector.text_images(upright, regions);
                // Lines on a page nearly always run the same way: each is read the way most of the
                // page's lines are classified. Only when the classifier is sure a line runs the other
                // way, or its reading is weak, is it read the other way as well, and that reading has
                // to be clearly better. (Deciding per line, a run of MRZ fillers - which reads about as
                // well upside down, < turned over being > - went whichever way was a shade more sure.)
                const auto ways = orientations(text_images);
                const auto upright_lines = std::ranges::count(ways, line_orientation::upright);
                const auto upside_down_lines = std::ranges::count(ways, line_orientation::upside_down);
                const bool page_upside_down = upside_down_lines > upright_lines;
                const auto against_page = page_upside_down ? line_orientation::upright : line_orientation::upside_down;

                std::vector<image> page_way;
                page_way.reserve(text_images.size());
                for (const auto &text_image: text_images) {
                    page_way.push_back(text_image);
                    if (page_upside_down) page_way.back().rotate(180);
                }
                const auto page_readings = recognizer.recognize(page_way);
                std::vector<image> other_way;
                std::vector<size_t> other_of;
                for (size_t i = 0; i < text_images.size(); ++i) {
                    if (ways[i] != against_page && page_readings[i].confidence >= RereadBelow) continue;
                    // One or two characters can't show which way up they are (an M upside down
                    // reads as W): they go the page's way.
                    if (!page_readings[i].text.empty() && page_readings[i].text.size() <= 2) continue;
                    other_way.push_back(text_images[i]);
                    if (!page_upside_down) other_way.back().rotate(180);
                    other_of.push_back(i);
                }
                const auto other_readings = other_way.empty() ? std::vector<ocr_recognizer::result>{} : recognizer.recognize(other_way);
                std::vector<const ocr_recognizer::result *> chosen(text_images.size());
                std::vector<bool> chosen_upside_down(text_images.size(), page_upside_down);
                for (size_t i = 0; i < text_images.size(); ++i) chosen[i] = &page_readings[i];
                for (size_t r = 0; r < other_of.size(); ++r) {
                    const size_t i = other_of[r];
                    if (static_cast<double>(other_readings[r].confidence) >= static_cast<double>(page_readings[i].confidence) + OtherWayMargin) {
                        chosen[i] = &other_readings[r];
                        chosen_upside_down[i] = !page_upside_down;
                    }
                }

                // An MRZ is tightly spaced: a double letter can merge into one (NAIDOO read NAIDO).
                // Lines that may be one (with a <, or 30 characters or more) are read again
                // stretched, and that reading is taken when it reads as an MRZ and is longer -
                // letters back - and about as confident. Ordinary text reads worse stretched.
                std::vector<image> dense;
                std::vector<size_t> dense_of;
                for (size_t i = 0; i < text_images.size(); ++i) {
                    const auto &text = chosen[i]->text;
                    if (text.find('<') == std::string::npos && text.size() < 30) continue;
                    dense.push_back(text_images[i]);
                    if (chosen_upside_down[i]) dense.back().rotate(180);
                    dense_of.push_back(i);
                }
                const auto dense_readings = dense.empty() ? std::vector<ocr_recognizer::result>{}
                                                          : recognizer.recognize(dense, DenseLineStretch);
                for (size_t r = 0; r < dense_of.size(); ++r) {
                    const size_t i = dense_of[r];
                    const auto &stretched = dense_readings[r].text, &first = chosen[i]->text;
                    if (stretched.find("<<") == std::string::npos) continue; // not an MRZ after all
                    const bool newly_mrz = first.find("<<") == std::string::npos;
                    if ((newly_mrz || stretched.size() > first.size()) &&
                        static_cast<double>(dense_readings[r].confidence) >= static_cast<double>(chosen[i]->confidence) - 5)
                        chosen[i] = &dense_readings[r];
                }

                for (size_t i = 0; i < regions.size(); ++i) {
                    const auto &recognized = *chosen[i];
                    if (recognized.text.empty() || recognized.confidence < minimum_confidence) continue;
                    lines_.push_back({
                        recognized.text, recognized.confidence,
                        unturned(static_cast<rect>(regions[i]), rotation_, input.size())
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
            // A box on the image turned by rotation (as image::rotate), back on the image as given.
            static rect unturned(const rect &box, const int rotation, const size_i &given) {
                const double w = given.width(), h = given.height();
                switch (rotation) {
                    case 90: // turned clockwise: (x, y) went to (h - y, x)
                        return {box.top(), h - box.right(), box.height(), box.width()};
                    case 180:
                        return {w - box.right(), h - box.bottom(), box.width(), box.height()};
                    case 270: // turned anticlockwise: (x, y) went to (y, w - x)
                        return {w - box.bottom(), box.left(), box.height(), box.width()};
                    default:
                        return box;
                }
            }

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
            bool auto_rotate = true;
            int rotation_ = 0;
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

    std::vector<ocr_recognizer::result> ocr_recognizer::recognize(const std::vector<image> &inputs, const double stretch) const {
        return impl->recognize(inputs, stretch);
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

    void ocr::set_auto_rotate(const bool auto_rotate) {
        impl->set_auto_rotate(auto_rotate);
    }

    int ocr::rotation() const {
        return impl->rotation();
    }

    int ocr::upright_rotation(const image &input) {
        static thread_local const std::unique_ptr<onnx> classifier = []() -> std::unique_ptr<onnx> {
            try {
                return std::make_unique<onnx>("paddle_rotate.onnx", false);
            } catch (const std::exception &) {
                return nullptr; // not installed (an older simply-cpp-models): taken as upright
            }
        }();
        if (!classifier || input.empty()) return 0;
        // The model sees the centre 224 x 224 pixels with the shorter side scaled to 256,
        // ImageNet-normalised. Its class k means the image is turned k quarters from upright.
        const auto size = input.size();
        const double scale = 256.0 / std::min(size.width(), size.height());
        const auto scaled = input.resized({
            std::max(224, static_cast<int>(std::lround(size.width() * scale))),
            std::max(224, static_cast<int>(std::lround(size.height() * scale)))
        });
        const auto scaled_size = scaled.size();
        const auto centre = scaled.cropped(rect_i{
            (scaled_size.width() - 224) / 2, (scaled_size.height() - 224) / 2, 224, 224
        });
        const auto outputs = classifier->process_image(centre, impl::DetectionScale, impl::DetectionMean);
        if (outputs.empty() || outputs.front().shape.size() != 2 || outputs.front().shape[1] != 4) return 0;
        const float *scores = outputs.front().data;
        const auto quarters = static_cast<int>(std::max_element(scores, scores + 4) - scores);
        return (4 - quarters) % 4 * 90;
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
