#include "yolo.h"
#include "image.h"
#include "onnx.h"
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace sc
{
    namespace impl
    {
        const std::vector<std::string> kCocoNames = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        };

        class detection
        {
        public:
            detection(const float* data, const image& img)
                : type(static_cast<int>(data[5])),
                  confidence(data[4], 1),
                  box(rect::ltrb(data[0], data[1], data[2], data[3]))
            {
                box -= img.padding();
                box /= img.cropped_size();
            }

            [[nodiscard]] nlohmann::ordered_json json() const
            {
                nlohmann::ordered_json result;
                result.push_back(static_cast<double>(percent{box.left() + box.width() / 2}));
                result.push_back(static_cast<double>(percent{box.top() + box.height() / 2}));
                result.push_back(static_cast<double>(percent{box.width()}));
                result.push_back(static_cast<double>(percent{box.height()}));
                result.push_back(static_cast<double>(confidence));
                result.push_back(type);
                return result;
            }

            [[nodiscard]] std::string object_name() const
            {
                return (type < 0 || type >= static_cast<int>(kCocoNames.size())) ? "unknown" : kCocoNames[type];
            }

            int type;
            percent confidence;
            rect box;
        };

        class yolo_impl
        {
        public:
            explicit yolo_impl(const std::string& model) : detector(model)
            {
                if (!detector.yolo26_size()) throw std::runtime_error("Model is not yolo26 compatible");
            }

            void set_threshold(const double threshold)
            {
                threshold_ = {threshold, 2};
            }

            void run(const std::string& image_filename)
            {
                detections.clear();
                original = std::make_unique<image>(image_filename);
                image input{*original};
                if (const auto model_size = detector.image_size(); model_size.width() > 0 && model_size.height() > 0)
                    input.snap_to_size(0, {model_size});
                const auto data = detector.process_image(input).front();
                for (int i = 0; i < detector.yolo26_size(); ++i) detections.emplace_back(data + i * 6, input);
            }

            [[nodiscard]] nlohmann::ordered_json json() const
            {
                nlohmann::ordered_json result;
                for (const auto& detection : detections)
                    if (detection.confidence >= threshold_) result.push_back(detection.json());
                return result;
            }

            [[nodiscard]] image annotated() const
            {
                if (!original) throw std::runtime_error("No image has been detected");
                image result{*original};
                for (const auto& detection : detections)
                {
                    if (detection.confidence < threshold_) continue;
                    const auto box = detection.box * result.size();
                    result.rect(box);

                    const auto label = detection.object_name() + " (" + detection.confidence + ")";
                    const auto top = box.top() < 25 ? box.top() + 25 : box.top() - 5;
                    result.text(label, {static_cast<int>(box.left()), static_cast<int>(top)});
                }
                return result;
            }

            onnx detector;
            std::unique_ptr<image> original;
            std::vector<detection> detections;

        private:
            percent threshold_{0.3f, 2};
        };
    }

    yolo::yolo(const std::string& model) : impl(new impl::yolo_impl(model))
    {
    }

    yolo::~yolo()
    {
        delete impl;
    }

    yolo::operator std::string() const
    {
        return impl->json().dump();
    }

    yolo::operator nlohmann::ordered_json() const
    {
        return impl->json();
    }

    std::ostream& operator<<(std::ostream& lhs, const yolo& rhs)
    {
        return lhs << static_cast<std::string>(rhs);
    }

    void yolo::set_threshold(const double threshold)
    {
        impl->set_threshold(threshold);
    }

    void yolo::detect(const std::filesystem::path& image_path) const
    {
        detect(image_path.string());
    }

    void yolo::detect(const std::string& image_filename) const
    {
        impl->run(image_filename);
    }

    bool yolo::display(const int timeout) const
    {
        return impl->annotated().show(timeout);
    }
} // namespace sc
