#pragma once

#include <ostream>
#include <string>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace sc {
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

        bool display(int timeout = 0) const;

        nlohmann::ordered_json to_json();

    private:
        impl::yolo_impl *impl;
    };
} // sc
