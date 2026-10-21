#pragma once

#include <cstddef>
#include <filesystem>
#include <image.h>
#include <nlohmann/json.hpp>
#include <ostream>
#include <percent.h>
#include <rect.h>
#include <memory>
#include <string>
#include <vector>

const std::string DETECTION_MODEL = "paddle_det_monkt_5";
const std::string RECOGNITION_MODEL = "paddle_rec_monkt_latin";

namespace sc {
    namespace impl {
        class ocr_detector_impl;
        class ocr_recognizer_impl;
        class ocr_impl;
    }

    class ocr_detector {
    public:
        explicit ocr_detector(const std::string &model = DETECTION_MODEL);

        ~ocr_detector();

        ocr_detector(ocr_detector &&) noexcept;

        ocr_detector &operator=(ocr_detector &&) noexcept;

        /// Sets the text-map threshold as a percentage from 0 to 100.
        void set_threshold(double threshold);

        /// Sets DBSCAN's minimum neighbor count for angle and height clusters.
        void set_minimum_neighbors(std::size_t minimum_neighbors);

        /// Sets DBSCAN's maximum height distance in detector-map pixels.
        void set_maximum_height_difference(double maximum_height_difference);

        /// Detects text regions in the source image, in reading order.
        [[nodiscard]] std::vector<rotated_rect> detect(const std::filesystem::path &image_path) const;
        [[nodiscard]] std::vector<rotated_rect> detect(const image &input) const;

        /// Crops and deskews detected text regions from the source image.
        [[nodiscard]] std::vector<image> text_images(const image &input, const std::vector<rotated_rect> &regions) const;

        /// Detects, crops, and deskews text regions from the source image.
        [[nodiscard]] std::vector<image> text_images(const std::filesystem::path &image_path) const;
        [[nodiscard]] std::vector<image> text_images(const image &input) const;

    private:
        std::unique_ptr<impl::ocr_detector_impl> impl;
    };

    class ocr_recognizer {
    public:
        struct result {
            std::string text;
            percent confidence;
        };

        explicit ocr_recognizer(const std::string &model = RECOGNITION_MODEL, const std::filesystem::path &dictionary = {});

        ~ocr_recognizer();

        ocr_recognizer(ocr_recognizer &&) noexcept;

        ocr_recognizer &operator=(ocr_recognizer &&) noexcept;

        /// Sets the minimum recognition confidence, as a percentage from 0 to 100, for batch results.
        void set_threshold(double threshold);

        /// Recognises one cropped text image.
        [[nodiscard]] result recognize(const image &input) const;

        /// Recognises cropped text images in a batch.
        [[nodiscard]] std::vector<result> recognize(const std::vector<image> &inputs) const;

        /// Recognises cropped text images stretched horizontally by stretch first: for tightly
        /// spaced text such as an MRZ, whose double letters can otherwise merge into one.
        [[nodiscard]] std::vector<result> recognize(const std::vector<image> &inputs, double stretch) const;

    private:
        std::unique_ptr<impl::ocr_recognizer_impl> impl;
    };

    class ocr {
    public:
        struct line {
            std::string text;
            percent confidence;
            rect box;
        };

        explicit ocr(const std::string &detection_model = DETECTION_MODEL,
                     const std::string &recognition_model = RECOGNITION_MODEL,
                     const std::filesystem::path &dictionary = {});

        ~ocr();

        ocr(ocr &&) noexcept;

        ocr &operator=(ocr &&) noexcept;

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const ocr &rhs);

        /// Sets the detector text-map threshold as a percentage from 0 to 100.
        void set_threshold(double threshold);

        /// Sets DBSCAN's minimum neighbor count for angle and height clusters.
        void set_minimum_neighbors(std::size_t minimum_neighbors);

        /// Sets DBSCAN's maximum height distance in detector-map pixels.
        void set_maximum_height_difference(double maximum_height_difference);

        /// Reads the image's text, keeping the lines read with at least minimum_confidence (a
        /// percentage; std::invalid_argument outside 0 to 100, or for an empty image).
        void detect(const std::filesystem::path &image_path, double minimum_confidence = 85) const;
        void detect(const image &input, double minimum_confidence = 85) const;

        /// Whether detect() first turns the image upright with the document orientation
        /// classifier (on by default). Lines' boxes are always in the given image's coordinates.
        void set_auto_rotate(bool auto_rotate);

        /// How far detect() turned the last image (0, 90, 180 or 270 degrees, as image::rotate).
        [[nodiscard]] int rotation() const;

        /// How far to turn an image (0, 90, 180 or 270 degrees, as image::rotate) for its text to
        /// be upright, from PaddleOCR's document orientation classifier; 0 when it isn't installed.
        [[nodiscard]] static int upright_rotation(const image &input);

        [[nodiscard]] const std::vector<line> &lines() const;

        [[nodiscard]] std::string text() const;

        /// The lines found, each boxed, on a copy of the image read last: to save or draw on.
        [[nodiscard]] image annotated() const;

        [[nodiscard]] nlohmann::ordered_json to_json() const;

    private:
        std::unique_ptr<impl::ocr_impl> impl;
    };
} // namespace sc
