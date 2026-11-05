#pragma once
#include <memory>
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
        /// gpu: the CUDA device to run on, from 0 (see onnx::gpus()).
        explicit facedetector(int gpu = 0);

        ~facedetector();

        facedetector(facedetector &&) noexcept;

        facedetector &operator=(facedetector &&) noexcept;

        void set_threshold(double threshold);

        void set_max_faces(int max_faces);

        [[nodiscard]] std::vector<face> detect(const std::filesystem::path &image_path,
                                                bool include_face_images = true) const;

        [[nodiscard]] std::vector<face> detect(const image &input, bool include_face_images = true) const;

        /// The faces found, boxed with their landmarks, on a copy of the image read last: to save or draw on.
        [[nodiscard]] image annotated() const;

        /// Where the faces the last detect() found are, in its image's own pixels (not the detector's
        /// letterboxed copy), in the order detect() gave them: e.g. to tell a portrait from a face
        /// far away by its height. None before a detect().
        [[nodiscard]] std::vector<rect> boxes() const;

    private:
        std::unique_ptr<impl::face_impl> impl;
    };
} // sc
