#include "onnx.h"
#include "image.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <sc.h>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using val = Ort::Value;

namespace sc {
    namespace impl {
        class onnx_impl {
        public:
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            onnx_impl(const std::string &Model) {
#ifdef __APPLE__
                // Add CoreML provider
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0));
#else
                OrtCUDAProviderOptions cuda_options{};
                options.AppendExecutionProvider_CUDA(cuda_options);
                options.SetIntraOpNumThreads(4);
                options.SetInterOpNumThreads(1);
#endif
                options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                session = new Ort::Session(env, Model.c_str(), options);
                for (const auto &val: session->GetInputs())
                    inputs.emplace_back(val.GetName().c_str(), val.TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
                for (const auto &val: session->GetOutputs())
                    outputs.emplace_back(val.GetName().c_str(), val.TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
                for (const auto &key: inputs) input_names.push_back(key.first.c_str());
                for (const auto &key: outputs) output_names.push_back(key.first.c_str());
            }

            ~onnx_impl() { delete session; }

            std::vector<const float *> run(const val &inputTensor) {
                results = session->Run(run_options,
                                       input_names.data(),
                                       &inputTensor,
                                       input_names.size(),
                                       output_names.data(),
                                       output_names.size());
                std::vector<const float *> data;
                data.reserve(results.size());
                for (auto &result: results) data.push_back(result.GetTensorData<float>());
                return data;
            }

            void show_shapes() {
                std::cout << "Input count: " << inputs.size() << "\n";
                std::cout << "Output count: " << outputs.size() << "\n";

                for (const auto &[name, shape]: inputs) {
                    std::cout << "Input name: " << name;
                    std::cout << "; shape: [ ";
                    for (const auto &s: shape) std::cout << s << " ";
                    std::cout << "]\n";
                }
                for (const auto &[name, shape]: outputs) {
                    std::cout << "Output name: " << name;
                    std::cout << "; shape: [ ";
                    for (const auto &s: shape) std::cout << s << " ";
                    std::cout << "]\n";
                }
            }

            [[nodiscard]] int yolo26_size() const {
                if (outputs.size() != 1) return 0;
                auto &shape = outputs.front().second;
                if (shape.size() != 3) return 0;
                if (shape[2] != 6) return 0;
                // Theoretically batch size could be more...
                if (shape[0] != 1) return 0;
                return static_cast<int>(shape[1]);
            }

            [[nodiscard]] size_i image_size() const {
                if (inputs.size() != 1) return {};
                auto input_shape = inputs[0].second;
                if (input_shape.size() != 4) return {};
                // Only handle batches of 1 for now...
                if (input_shape[0] != 1) return {};
                // Only handling RGB
                if (input_shape[1] != 3) return {};
                return {input_shape[2], input_shape[3]};
            }

        private:
#ifdef NDEBUG
            Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "simply-cpp-onnx"};
#else
            Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "simply-cpp-onnx"};
#endif
            Ort::SessionOptions options;
            Ort::RunOptions run_options{nullptr};
            Ort::Session *session;
            std::vector<val> results;

            std::vector<std::pair<std::string, std::vector<int64_t> > > inputs;
            std::vector<std::pair<std::string, std::vector<int64_t> > > outputs;
            std::vector<const char *> input_names;
            std::vector<const char *> output_names;
        };
    }

    onnx::onnx(const std::string &model) : impl(new impl::onnx_impl(model)) {
    }

    onnx::~onnx() { delete impl; }

    std::vector<const float *> onnx::process_image(const image &img) {
        img.generate_blob(1.0 / 255.0, 127.5, true);
        const val input = val::CreateTensor<float>(impl->memInfo,
                                                   img.blob(), img.blob_size(),
                                                   img.blob_shape(), img.blob_shape_size());
        return impl->run(input);
    }

    int onnx::yolo26_size() const {
        return impl->yolo26_size();
    }

    size_i onnx::image_size() const {
        return impl->image_size();
    }

    void onnx::show_shapes() const {
        impl->show_shapes();
    }

    void onnx::show_providers() {
        for (const auto &provider: Ort::GetAvailableProviders()) std::cout << provider << '\n';
    }
} // sc
