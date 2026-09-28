#pragma once

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

        explicit ocr(const std::string &detection_model = "paddle_det_s",
                      const std::string &recognition_model = "paddle_rec_s",
                      const std::filesystem::path &dictionary = {});

        ~ocr();

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const ocr &rhs);

        /// Sets the text-map threshold as a percentage from 0 to 100.
        void set_threshold(double threshold);

        void detect(const std::filesystem::path &image_path) const;

        void detect(const image &input) const;

        [[nodiscard]] const std::vector<line> &lines() const;

        [[nodiscard]] std::string text() const;

        [[nodiscard]] bool display(int timeout = 0) const;

        [[nodiscard]] nlohmann::ordered_json to_json() const;

    private:
        impl::ocr_impl *impl;
    };
} // namespace sc
