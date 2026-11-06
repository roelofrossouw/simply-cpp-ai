#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>
#include "image.h"

namespace sc {
    namespace impl {
        class onnx_impl;
    }

    class output {
    public:
        output(std::vector<int64_t> s, const float *d) : shape(s) {
            data = const_cast<float *>(d);
        }

        explicit operator std::string() const {
            std::string result = "[ ";
            for (const auto &dimension: shape) result += std::to_string(dimension) + " ";
            result += ']';
            return result;
        }

        // The number of values in the output: the product of its dimensions (0 for no shape).
        [[nodiscard]] int data_size() const {
            if (shape.empty()) return 0;
            return static_cast<int>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>()));
        }

        [[nodiscard]] int max_element() const {
            return static_cast<int>(std::max_element(data, data + data_size()) - data);
        }

        friend std::ostream &operator <<(std::ostream &os, const output &rhs) {
            return os << static_cast<std::string>(rhs);
        }

        std::vector<int64_t> shape;
        float *data;
    };

    class onnx {
    public:
        /// gpu: the CUDA device to run on, from 0 (see gpus()); throws std::invalid_argument for one
        /// that isn't there. Without CUDA (and on Apple platforms) it isn't used.
        explicit onnx(const std::string &model, bool use_metal = true, int gpu = 0);

        ~onnx();

        onnx(onnx &&) noexcept;

        onnx &operator=(onnx &&) noexcept;

        /// Runs one inference for an RGB image, given to the model as (pixel - 127.5) / 255.
        std::vector<output> process_image(const image &img) const;

        /// Runs one inference for an RGB image, given to the model as (pixel - mean) * scale.
        std::vector<output> process_image(const image &img, double scale, double mean) const;

        /// Runs one inference for equally tall RGB images, padding them to a common width.
        std::vector<output> process_images(const std::vector<image> &images) const;

        /// The same, with each image given to the model as (pixel - mean) * scale.
        std::vector<output> process_images(const std::vector<image> &images, double scale, double mean) const;

        /// The same, padded to at least width: a long list read a few at a time, each as if in one batch
        /// as wide as the widest of all (a model can read an image differently with more padding).
        std::vector<output> process_images(const std::vector<image> &images, double scale, double mean, int width) const;

        [[nodiscard]] int yolo26_size() const;

        // The input image size, width by height, for a model taking one RGB image ([1, 3, height,
        // width]); a dynamic dimension is not positive. Empty for any other kind of input.
        [[nodiscard]] size_i image_size() const;

        void show_shapes() const;

        static void show_providers();

        // Whether models run accelerated: CoreML on Apple platforms, or CUDA where its provider
        // and runtime are available. Without, they run on the CPU (several times slower).
        [[nodiscard]] static bool accelerated();

        /// The CUDA devices models can run on: 0 without CUDA (always on Apple platforms).
        [[nodiscard]] static int gpus();

        /// The most GPU memory, in bytes, each model loaded after this call may take for its work
        /// (ONNX Runtime's CUDA arena; the model's weights come on top); 0, the default: no limit.
        /// For a program sharing its GPU with others. A run that needs more fails with an exception.
        static void gpu_memory_limit(std::size_t bytes);
        [[nodiscard]] static std::size_t gpu_memory_limit();

    private:
        static std::atomic<std::size_t> &gpu_memory_limit_setting();

        std::unique_ptr<impl::onnx_impl> impl;
    };
} // sc
