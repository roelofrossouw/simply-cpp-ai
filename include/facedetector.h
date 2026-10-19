#pragma once
#include <string>
#include <filesystem>
#include "face.h"

namespace sc {
    class image;

    namespace impl {
        class face_impl;
    }

    class facedetector {
    public:
        facedetector();

        ~facedetector();


        void set_threshold(double threshold);

        void set_max_faces(int max_faces);

        [[nodiscard]] std::vector<face> detect(const std::filesystem::path &image_path,
                                                bool include_face_images = true) const;

        [[nodiscard]] std::vector<face> detect(const image &input, bool include_face_images = true) const;

        /// The faces found, boxed with their landmarks, on a copy of the image read last: to save or draw on.

        [[nodiscard]] image annotated() const;

    private:
        impl::face_impl *impl;
    };
} // sc
