#pragma once

#include <chrono>
#include <ostream>
#include <string>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace sc {
    class image;
    class video;

    namespace impl {
        class yolo_impl;
    }

    class yolo {
    public:
        /// gpu: the CUDA device to run on, from 0 (see onnx::gpus()).
        explicit yolo(const std::string &model, int gpu = 0);

        ~yolo();

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const yolo &rhs);

        void set_threshold(double threshold);

        void detect(const std::filesystem::path &image_path) const;
        void detect(const image &input) const;

        /// What detect(video) gives: what an image would (detail::best), or a summary (detail::full).
        enum class detail { best, full };

        /// Runs over a video a frame every so often (sc-image's sc::video::next(every)).
        ///
        /// detail::best gives what to_json() gives for an image, so a video and a photo read the same:
        /// the detections, [centre x %, centre y %, width %, height %, confidence %, type], here each
        /// kind of object's surest one in the whole video, most certain first:
        ///
        ///     [[38.14, 56.46, 75.2, 84.97, 93.6, 0], [37.64, 73.66, 11.68, 7.25, 68.8, 67]]
        ///
        /// detail::full sums up, for each kind of object: how many frames it was in (count), how sure
        /// it was at most (confidence, %), that frame's number and time (ms), and all that frame's
        /// detections, of every kind, as for an image. Most certain first:
        ///
        ///     {"frames": 15, "every": 1000, "objects": {
        ///         "person": {"count": 15, "confidence": 93.6, "frame": 45, "time": 2987,
        ///                    "detections": [[38.14, 56.46, 75.2, 84.97, 93.6, 0], ...]}, ...}}
        ///
        /// Afterwards to_json() and annotated() are the last frame's.
        [[nodiscard]] nlohmann::ordered_json detect(video &clip, std::chrono::milliseconds every = std::chrono::seconds(1),
                                                    detail amount = detail::best) const;

        /// The objects found, each boxed and labelled, on a copy of the image read last: to save or draw on.

        [[nodiscard]] image annotated() const;

        nlohmann::ordered_json to_json() const;

        // The COCO name of a detection's type (the last value of each detection in to_json():
        // [centre x %, centre y %, width %, height %, confidence %, type]), or "unknown".
        [[nodiscard]] static std::string object_name(int type);

    private:
        impl::yolo_impl *impl;
    };
} // sc
