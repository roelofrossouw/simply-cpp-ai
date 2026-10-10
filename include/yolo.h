#pragma once

#include <ostream>
#include <string>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace sc {
    class image;

    namespace impl {
        class yolo_impl;
    }

    class yolo {
    public:
        yolo(const std::string &model);

        ~yolo();

        operator std::string() const;

        operator nlohmann::ordered_json() const;

        friend std::ostream &operator<<(std::ostream &lhs, const yolo &rhs);

        void set_threshold(double threshold);

        void detect(const std::filesystem::path &image_path) const;
        void detect(const image &input) const;

        bool display(int timeout = 0) const;

        nlohmann::ordered_json to_json() const;

        // The COCO name of a detection's type (the last value of each detection in to_json():
        // [centre x %, centre y %, width %, height %, confidence %, type]), or "unknown".
        [[nodiscard]] static std::string object_name(int type);

    private:
        impl::yolo_impl *impl;
    };
} // sc
