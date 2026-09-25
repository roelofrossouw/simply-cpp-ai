#pragma once
#include <string>
#include <filesystem>
#include <opencv.h>
#include <nlohmann/json.hpp>

namespace sc {
    namespace impl {
        class face_impl;
    }

    struct face {
        static constexpr int feature_size = 512;

        face(const float *feats);

        face(const face &copy);

        void setFeatures(const float *feats);

        // same image      1.00
        // same person     0.65 - 0.90
        // different       0.10 - 0.40
        [[nodiscard]] percent similarity(const face &rhs) const;

        percent operator^(const face &rhs) const;

        void normalize();

        float features[feature_size]{};
    };

    class facedetector {
    public:
        facedetector();

        ~facedetector();

        friend std::ostream &operator<<(std::ostream &lhs, const facedetector &rhs);

        void set_threshold(double threshold);

        void set_max_faces(int max_faces);

        void detect(const std::filesystem::path &image_path) const;

        [[nodiscard]] bool display(int timeout = 0) const;

        [[nodiscard]] const std::vector<image> &faces() const;

        [[nodiscard]] const std::vector<face> &features() const;

    private:
        impl::face_impl *impl;
    };
} // sc
