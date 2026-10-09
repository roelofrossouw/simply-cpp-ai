#include "onnx.h"
#include "image.h"
#include <algorithm>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <sc.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#else
#include <dlfcn.h>
#endif
#include "config.h"
const std::filesystem::path default_model_dir = SIMPLY_CPP_MODEL_DIR;

using val = Ort::Value;
namespace fs = std::filesystem;

namespace sc {
    namespace impl {
#ifndef __APPLE__
        bool cuda_runtime_available() {
            constexpr const char *libraries[] = {
                "libcublasLt.so.13",
                "libcublasLt.so"
            };
            for (const auto *library: libraries) {
                void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
                if (handle) {
                    dlclose(handle);
                    return true;
                }
            }
            return false;
        }
#endif

        class onnx_impl {
        public:
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            onnx_impl(const std::string &Model, const bool use_metal) : model_filename(Model) {
                if (!fs::exists(model_filename)) {
                    if (fs::exists(model_filename.string() + ".onnx"))
                        model_filename += ".onnx";
                    else if (fs::exists(default_model_dir / model_filename))
                        model_filename = default_model_dir / model_filename;
                    else if (fs::exists((default_model_dir / model_filename).string() + ".onnx"))
                        model_filename = (default_model_dir / model_filename).string() + ".onnx";
                    else throw std::runtime_error("Model file not found. " + model_filename.string());
                }

#ifdef __APPLE__
                // Add CoreML provider
                if (use_metal) {
                    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0));
                }
#else
                const auto providers = Ort::GetAvailableProviders();
                const auto cuda = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
                if (cuda != providers.end() && cuda_runtime_available()) {
                    try {
                        OrtCUDAProviderOptions cuda_options{};
                        options.AppendExecutionProvider_CUDA(cuda_options);
                    } catch (const Ort::Exception &error) {
                        std::cerr << "CUDA execution provider unavailable: " << error.what() << '\n';
                    }
                } else if (cuda != providers.end()) {
                    std::cerr << "CUDA execution provider found, but libcublasLt is unavailable; "
                            "using the default provider\n(Consider installing cuda-toolkit-13-4 to enable GPU usage)";
                }
                options.SetIntraOpNumThreads(4);
                options.SetInterOpNumThreads(1);
#endif
                options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#ifndef NDEBUG
                sc::timer sw;
                std::cerr << "Loading model " << model_filename << std::endl;
#endif

                // options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                // options.SetIntraOpNumThreads(10);

                session = new Ort::Session(env, model_filename.string().c_str(), options);


                for (const auto &val: session->GetInputs())
                    inputs.emplace_back(val.GetName().c_str(), val.TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
                for (const auto &val: session->GetOutputs())
                    outputs.emplace_back(val.GetName().c_str(), val.TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
                for (const auto &key: inputs) input_names.push_back(key.first.c_str());
                for (const auto &key: outputs) output_names.push_back(key.first.c_str());
#ifndef NDEBUG
                std::cerr << "Loaded in " << sw << std::endl;
#endif
            }

            ~onnx_impl() { delete session; }

            std::vector<output> run(const val &inputTensor) {
                results = session->Run(run_options,
                                       input_names.data(),
                                       &inputTensor,
                                       input_names.size(),
                                       output_names.data(),
                                       output_names.size());
                std::vector<output> data;
                data.reserve(results.size());
                for (auto &result: results)
                    data.emplace_back(result.GetTensorTypeAndShapeInfo().GetShape(), result.GetTensorData<float>());

                return data;
            }

            std::vector<output> run_images(const std::vector<image> &images, const double scale, const double mean) {
                if (images.empty()) return {};
                if (inputs.size() != 1 || inputs.front().second.size() != 4 ||
                    inputs.front().second[1] != 3)
                    throw std::runtime_error{"Batched image inference requires one NCHW RGB model input"};

                const auto &model_shape = inputs.front().second;
                const auto model_batch = model_shape[0];
                if (model_batch > 0 && static_cast<size_t>(model_batch) != images.size())
                    throw std::invalid_argument{"Image batch size does not match the model's fixed batch dimension"};

                int height = model_shape[2] > 0 ? static_cast<int>(model_shape[2]) : 0;
                int width = model_shape[3] > 0 ? static_cast<int>(model_shape[3]) : 0;
                for (const auto &img: images) {
                    if (img.empty()) throw std::invalid_argument{"Image batch cannot contain empty images"};
                    img.generate_blob(scale, mean, true);
                    if (img.blob_shape_size() != 4 || img.blob_shape()[0] != 1 || img.blob_shape()[1] != 3)
                        throw std::invalid_argument{"Image blobs must have shape [1, 3, height, width]"};
                    const int image_height = static_cast<int>(img.blob_shape()[2]);
                    const int image_width = static_cast<int>(img.blob_shape()[3]);
                    if (model_shape[2] > 0 && image_height != model_shape[2])
                        throw std::invalid_argument{"Image height does not match the model's fixed input height"};
                    if (model_shape[3] > 0 && image_width > model_shape[3])
                        throw std::invalid_argument{"Image width exceeds the model's fixed input width"};
                    height = std::max(height, image_height);
                    width = std::max(width, image_width);
                }

                const std::vector<int64_t> shape{
                    model_batch > 0 ? model_batch : static_cast<int64_t>(images.size()),
                    3,
                    height,
                    width
                };
                const auto batch = static_cast<size_t>(shape[0]);
                const auto plane_size = static_cast<size_t>(height) * width;
                std::vector<float> batch_data(batch * 3 * plane_size, 0.0f);
                for (size_t image_index = 0; image_index < images.size(); ++image_index) {
                    const auto &img = images[image_index];
                    const int source_height = static_cast<int>(img.blob_shape()[2]);
                    const int source_width = static_cast<int>(img.blob_shape()[3]);
                    const auto *source = img.blob();
                    for (size_t channel = 0; channel < 3; ++channel) {
                        const auto *source_plane = source + channel * source_height * source_width;
                        auto *target_plane = batch_data.data() + (image_index * 3 + channel) * plane_size;
                        for (int row = 0; row < source_height; ++row)
                            std::copy_n(source_plane + row * source_width, source_width,
                                        target_plane + row * width);
                    }
                }

                const val input_tensor = val::CreateTensor<float>(
                    memInfo, batch_data.data(), batch_data.size(), shape.data(), shape.size());
                return run(input_tensor);
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
            fs::path model_filename;
        };
    }

    onnx::onnx(const std::string &model, const bool use_metal) : impl(new impl::onnx_impl(model, use_metal)) {
    }

    onnx::~onnx() { delete impl; }

    std::vector<output> onnx::process_image(const image &img) const {
        return process_image(img, 1.0 / 255.0, 127.5);
    }

    std::vector<output> onnx::process_image(const image &img, const double scale, const double mean) const {
        img.generate_blob(scale, mean, true);
        const val input = val::CreateTensor<float>(impl->memInfo,
                                                   img.blob(), img.blob_size(),
                                                   img.blob_shape(), img.blob_shape_size());
        return impl->run(input);
    }

    std::vector<output> onnx::process_images(const std::vector<image> &images) const {
        return process_images(images, 1.0 / 255.0, 127.5);
    }

    std::vector<output> onnx::process_images(const std::vector<image> &images, const double scale,
                                             const double mean) const {
        return impl->run_images(images, scale, mean);
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
