#pragma once

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
            int64_t size = 1;
            for (const auto &s: shape) size *= s;
            return static_cast<int>(size);
        }

        [[nodiscard]] int max_element() const {
            return static_cast<int>(std::max_element(data, data + data_size()) - data);
        }

        friend std::ostream &operator <<(std::ostream &os, const output &rhs) {
            os << "[";
            for (const auto s: rhs.shape) os << " " << s;
            return os << " ]";
        }

        std::vector<int64_t> shape;
        float *data;
    };

    class onnx {
    public:
        explicit onnx(const std::string &model, bool use_metal = true);

        ~onnx();

        /// Runs one inference for an RGB image, given to the model as (pixel - 127.5) / 255.
        std::vector<output> process_image(const image &img) const;

        /// Runs one inference for an RGB image, given to the model as (pixel - mean) * scale.
        std::vector<output> process_image(const image &img, double scale, double mean) const;

        /// Runs one inference for equally tall RGB images, padding them to a common width.
        std::vector<output> process_images(const std::vector<image> &images) const;

        /// The same, with each image given to the model as (pixel - mean) * scale.
        std::vector<output> process_images(const std::vector<image> &images, double scale, double mean) const;

        [[nodiscard]] int yolo26_size() const;

        // The input image size, width by height, for a model taking one RGB image ([1, 3, height,
        // width]); a dynamic dimension is not positive. Empty for any other kind of input.
        [[nodiscard]] size_i image_size() const;

        void show_shapes() const;

        static void show_providers();

    private:
        impl::onnx_impl *impl;
    };
} // sc
