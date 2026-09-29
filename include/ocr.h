#pragma once

#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <ostream>
#include <percent.h>
#include <rect.h>
#include <string>
#include <vector>

namespace sc {
    class image;

    namespace impl {
        class ocr_impl;
    }

    class ocr {
    public:
        struct line {
            std::string text;
            percent confidence;
            rect box;
        };

        explicit ocr(const std::string &detection_model = "paddle_det_monkt_5",
                      const std::string &recognition_model = "paddle_rec_monkt_latin",
                      const std::filesystem::path &dictionary = {});

        ~ocr();

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const ocr &rhs);

        /// Sets the text-map threshold as a percentage from 0 to 100.
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
