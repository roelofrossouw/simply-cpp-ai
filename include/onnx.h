#pragma once

#include <string>
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

        operator std::string() const {
            std::string result = "[ ";
            for (const auto &dimension: shape) result += std::to_string(dimension) + " ";
            result += ']';
            return result;
        }

        std::vector<int64_t> shape;
        float *data;
    };

    class onnx {
    public:
        explicit onnx(const std::string &model);

        ~onnx();

        std::vector<output> process_image(const image &img) const;

        [[nodiscard]] int yolo26_size() const;

        [[nodiscard]] size_i image_size() const;

        void show_shapes() const;

        static void show_providers();

    private:
        impl::onnx_impl *impl;
    };
} // sc
