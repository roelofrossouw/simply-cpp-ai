#pragma once

#include <string>
#include "image.h"

namespace sc
{
    namespace impl
    {
        class onnx_impl;
    }

    class onnx
    {
    public:
        onnx(const std::string& model);

        ~onnx();

        std::vector<const float*> process_image(const image& img);
        [[nodiscard]] int yolo26_size() const;
        [[nodiscard]] size_i image_size() const;
        void show_shapes() const;
        static void show_providers();

    private:
        impl::onnx_impl* impl;
    };
} // sc
