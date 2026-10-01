#pragma once

#include <cstddef>
#include <filesystem>
#include <image.h>
#include <nlohmann/json.hpp>
#include <ostream>
#include <percent.h>
#include <rect.h>
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

        /// Sets the text-map threshold as a percentage from 0 to 100.
        void set_threshold(double threshold);

        /// Sets DBSCAN's minimum neighbor count for angle and height clusters.
        void set_minimum_neighbors(std::size_t minimum_neighbors);

        /// Sets DBSCAN's maximum height distance in detector-map pixels.
        void set_maximum_height_difference(double maximum_height_difference);

        /// Detects and orders text regions in the source image.
        [[nodiscard]] std::vector<rotated_rect> detect(const std::filesystem::path &image_path) const;

        /// Detects and orders text regions in the source image.
        [[nodiscard]] std::vector<rotated_rect> detect(const image &input) const;

        /// Crops and deskews detected text regions from the source image.
        [[nodiscard]] std::vector<image> text_images(const image &input, const std::vector<rotated_rect> &regions) const;

        /// Detects, crops, and deskews text regions from the source image.
        [[nodiscard]] std::vector<image> text_images(const std::filesystem::path &image_path) const;

        /// Detects, crops, and deskews text regions from the source image.
        [[nodiscard]] std::vector<image> text_images(const image &input) const;

    private:
        impl::ocr_detector_impl *impl;
    };

    class ocr_recognizer {
    public:
        struct result {
            std::string text;
            percent confidence;
        };

        explicit ocr_recognizer(const std::string &model = RECOGNITION_MODEL, const std::filesystem::path &dictionary = {});

        ~ocr_recognizer();

        /// Recognises one cropped text image.
        [[nodiscard]] result recognize(const image &input) const;

        /// Recognises cropped text images in a batch.
        [[nodiscard]] std::vector<result> recognize(const std::vector<image> &inputs) const;

    private:
        impl::ocr_recognizer_impl *impl;
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

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const ocr &rhs);

        /// Sets the detector text-map threshold as a percentage from 0 to 100.
        void set_threshold(double threshold);

        /// Sets DBSCAN's minimum neighbor count for angle and height clusters.
        void set_minimum_neighbors(std::size_t minimum_neighbors);

        /// Sets DBSCAN's maximum height distance in detector-map pixels.
        void set_maximum_height_difference(double maximum_height_difference);

        /// Detects text with at least this recognition confidence, as a percentage from 0 to 100.
        void detect(const std::filesystem::path &image_path, double minimum_confidence = 0) const;

        /// Detects text with at least this recognition confidence, as a percentage from 0 to 100.
        void detect(const image &input, double minimum_confidence = 0) const;

        [[nodiscard]] const std::vector<line> &lines() const;

        [[nodiscard]] std::string text() const;

        [[nodiscard]] bool display(int timeout = 0) const;

        [[nodiscard]] nlohmann::ordered_json to_json() const;

    private:
        impl::ocr_impl *impl;
    };
} // namespace sc
