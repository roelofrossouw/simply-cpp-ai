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

        [[nodiscard]] int data_size() const {
            int64_t size(0);
            for (const auto &s: shape) size += s;
            return static_cast<int>(size);
        }

        [[nodiscard]] int max_element() const {
            return static_cast<int>(std::max_element(data, data + data_size()) - data);
        }

        std::vector<int64_t> shape;
        float *data;
    };

    class onnx {
    public:
        explicit onnx(const std::string &model);

        ~onnx();

        std::vector<output> process_image(const image &img) const;

        /// Runs one inference for equally tall RGB images, padding them to a common width.
        std::vector<output> process_images(const std::vector<image> &images) const;

        [[nodiscard]] int yolo26_size() const;

        [[nodiscard]] size_i image_size() const;

        void show_shapes() const;

        static void show_providers();

    private:
        impl::onnx_impl *impl;
    };
} // sc
