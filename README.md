# simply-cpp-ai

Wrappers for common AI tooling: ONNX, YOLO, and more.

The public API uses the `sc` namespace and builds on `simply-cpp` (sc-core) and `simply-cpp-image` (sc-image).

## Install

### Homebrew (macOS)

```bash
curl -fsSL https://apt.roelof.co.za/setup.sh | bash # taps roelofrossouw/sc - same command as the apt one below
brew install simply-cpp simply-cpp-image simply-cpp-ai
```

### apt (Ubuntu)

```bash
curl -fsSL https://apt.roelof.co.za/setup.sh | bash # registers the apt repo - same command as the brew one above
sudo apt -y install simply-cpp-dev simply-cpp-image-dev simply-cpp-ai-dev
```

### CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
        sc-ai
        GIT_REPOSITORY https://github.com/roelofrossouw/simply-cpp-ai.git
        GIT_TAG main # or a specific tag, e.g. v1.0.2, to stay stable
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sc-ai)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE sc::sc-ai)
```

sc-core and sc-image are fetched automatically as part of this if they aren't already available - no separate step needed.

### Git submodule

```bash
git submodule add https://github.com/roelofrossouw/simply-cpp-ai.git third_party/sc-ai
```

```cmake
add_subdirectory(third_party/sc-ai)
target_link_libraries(myapp PRIVATE sc::sc-ai)
```

## Dependencies

- **simply-cpp (sc-core) and simply-cpp-image (sc-image)** - real dependencies of sc-ai's CMake package. Neither the Homebrew formula nor the apt package currently pulls them in automatically, so install all three explicitly (see above). FetchContent and the git submodule route fetch/build them automatically instead.
- **nlohmann_json** - `nlohmann-json3-dev` on apt, `nlohmann-json` on brew; installed automatically if missing when building from source.
- **ONNX Runtime** - `simply-cpp-onnxruntime` on apt (a repackaged build of ONNX Runtime's own GPU/CUDA release - a matching NVIDIA/CUDA setup needs to already be present), `onnxruntime` on brew; installed automatically if missing when building from source.
- **ONNX model files** - a separate `simply-cpp-models` package supplies the actual `.onnx` model weights; installed automatically if missing when building from source.

Both of the above come from the same Homebrew tap / apt repo as `simply-cpp` itself, and CMake registers that tap/repo automatically the first time it needs to - nothing to set up by hand first, even on a completely fresh machine.

## Usage

`sc-ai`'s own CMake package doesn't yet declare its dependency on `sc-image`, so when consuming an installed package, find that one explicitly too:

```cmake
find_package(sc-image CONFIG REQUIRED)
find_package(sc-ai CONFIG REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE sc::sc-ai)
```

(FetchContent and the git submodule route above don't need this - they resolve it automatically.)

```cpp
#include <yolo.h>
#include <iostream>

int main() {
    sc::yolo yolo("yolo26n.onnx");
    yolo.set_threshold(50);
    yolo.detect("photo.jpg");
    std::cout << yolo << std::endl;
}
```

`sc::onnx` (`<onnx.h>`) is the lower-level wrapper `sc::yolo` is built on, for running other ONNX models directly against `sc::image` data.

PaddleOCR is available through `sc::ocr`. It detects text regions, recognises each line, and reports text, confidence, and its bounding box in the source image:

```cpp
#include <ocr.h>

sc::ocr reader; // Uses paddle_det_s and paddle_rec_s from simply-cpp-models
reader.detect("document.jpg");
std::cout << reader.text() << std::endl;
std::cout << reader.to_json().dump(2) << std::endl;
```

Use the optional constructor arguments to select detection and recognition models, and an optional third path for a custom recognition dictionary. `reader.lines()` returns the recognised line records directly; `reader.display()` shows the source image with text boxes.

`sc::identity` passes the original image to the same PaddleOCR engine and bundled English models, then parses its output for document fields. It does not add preprocessing or retries. Tesseract is no longer required; identity recognition remains English-only.

Pass `true` as the second argument to also detect face embeddings from the original image. The embeddings are available through `faces()` and are included in JSON only when face detection was requested:

```cpp
sc::identity document{"id.jpg", true};
for (const auto &detected_face: document.faces()) {
    save_for_matching(detected_face.to_json());
}
```

### Face detection and reusable face data

`sc::facedetector` returns detected faces as `std::vector<sc::face>`. Each `sc::face` owns normalized 512-value features and can optionally retain its aligned face image:

```cpp
#include <facedetector.h>

sc::facedetector detector;
const auto faces = detector.detect("photo.jpg"); // Keep aligned face images (default)
const auto embeddings = detector.detect("photo.jpg", false); // Feature-only faces

if (!embeddings.empty()) {
    const nlohmann::ordered_json json = embeddings.front().to_json();
    embeddings.front().save("face.json");
    const auto loaded = sc::face::load("face.json");
}
```

Face JSON and file persistence store only the feature vector; the optional image is transient and omitted. This keeps serialized records lightweight and suitable for a database. `face::from_json()` and `face::load()` restore feature-only faces. Similarity is available through `similarity()` or `operator^`.

## Demo

`sc-ai-demo` is installed with the runtime package (`simply-cpp-ai`), so you can
check that ONNX Runtime and the models work without the `-dev` package. It runs
YOLO object detection on an image file, or, with no argument, on a small
rendered image. That finds nothing, but proves everything loads and runs:

```bash
sc-ai-demo
sc-ai-demo street.jpg
```

Its source is `examples/sc-ai-demo.cpp`; the code below is copied from it at
configure time, so it always matches code that compiles:

<!-- sc-example: examples/sc-ai-demo.cpp -->
```cpp
sc::timer sw;
const sc::yolo yolo{"yolo26n.onnx"};
std::cout << "Model loaded after " << sw << '\n';

yolo.detect(std::filesystem::path{source});
std::cout << source << ": " << yolo << '\n';
std::cout << "Done after " << sw << '\n';
```
<!-- /sc-example -->

## Requirements

- CMake 3.22 or newer
- A C++20 compiler
- See Dependencies above for OpenCV (via sc-image), ONNX Runtime, and nlohmann_json

## Building and testing

```bash
cmake -B build -S .
cmake --build build -j
ctest --test-dir build --output-on-failure
```
