#include "face.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace sc {
    face::face(const float *features) {
        set_features(features);
    }

    face::face(const float *features, const image &face_image) : image_(face_image) {
        if (face_image.empty()) throw std::invalid_argument{"Face image cannot be empty"};
        set_features(features);
    }

    face::face(const feature_vector &features) : face(features.data()) {
    }

    face::face(const feature_vector &features, const image &face_image) : face(features.data(), face_image) {
    }

    const face::feature_vector &face::features() const noexcept {
        return features_;
    }

    bool face::has_image() const noexcept {
        return image_.has_value();
    }

    const image *face::face_image() const noexcept {
        return image_ ? &*image_ : nullptr;
    }

    percent face::similarity(const face &rhs) const {
        return {std::inner_product(features_.begin(), features_.end(), rhs.features_.begin(), 0.0f), 0};
    }

    percent face::operator^(const face &rhs) const {
        return similarity(rhs);
    }

    nlohmann::ordered_json face::to_json() const {
        return nlohmann::ordered_json{{"version", 1}, {"features", features_}};
    }

    face face::from_json(const nlohmann::ordered_json &json) {
        if (!json.is_object() || json.value("version", 0) != 1)
            throw std::invalid_argument{"Unsupported face JSON format"};
        if (!json.contains("features") || !json["features"].is_array()
            || json["features"].size() != feature_size)
            throw std::invalid_argument{"Face JSON must contain exactly 512 feature values"};

        feature_vector features;
        for (size_t i = 0; i < feature_size; ++i) {
            if (!json["features"][i].is_number())
                throw std::invalid_argument{"Face features must be numeric"};
            features[i] = json["features"][i].get<float>();
        }
        return face{features};
    }

    void face::save(const std::filesystem::path &path) const {
        std::ofstream file{path};
        if (!file) throw std::runtime_error{"Could not open face file for writing: " + path.string()};
        file << to_json().dump(2) << '\n';
        if (!file) throw std::runtime_error{"Could not write face file: " + path.string()};
    }

    face face::load(const std::filesystem::path &path) {
        std::ifstream file{path};
        if (!file) throw std::runtime_error{"Could not open face file for reading: " + path.string()};
        nlohmann::ordered_json json;
        file >> json;
        return from_json(json);
    }

    void face::set_features(const float *features) {
        if (!features) throw std::invalid_argument{"Face features cannot be null"};
        std::copy_n(features, feature_size, features_.begin());
        double norm_squared = 0.0;
        for (const float feature: features_) {
            if (!std::isfinite(feature)) throw std::invalid_argument{"Face features must be finite"};
            norm_squared += static_cast<double>(feature) * feature;
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 0.0f)
            throw std::invalid_argument{"Face features must have a finite, non-zero norm"};
        const double norm = std::sqrt(norm_squared);
        for (float &feature: features_) feature = static_cast<float>(feature / norm);
    }
} // namespace sc
