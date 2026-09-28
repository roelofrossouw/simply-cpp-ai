#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <opencv.h>
#include <nlohmann/json.hpp>

namespace sc {
    class face {
    public:
        static constexpr size_t feature_size = 512;
        using feature_vector = std::array<float, feature_size>;

        explicit face(const float *features);
        face(const float *features, const image &face_image);
        explicit face(const feature_vector &features);
        face(const feature_vector &features, const image &face_image);

        [[nodiscard]] const feature_vector &features() const noexcept;
        [[nodiscard]] bool has_image() const noexcept;
        [[nodiscard]] const image *face_image() const noexcept;

        [[nodiscard]] percent similarity(const face &rhs) const;
        [[nodiscard]] percent operator^(const face &rhs) const;

        [[nodiscard]] nlohmann::ordered_json to_json() const;
        [[nodiscard]] static face from_json(const nlohmann::ordered_json &json);
        void save(const std::filesystem::path &path) const;
        [[nodiscard]] static face load(const std::filesystem::path &path);

    private:
        feature_vector features_{};
        std::optional<image> image_;

        void set_features(const float *features);
    };
} // namespace sc
