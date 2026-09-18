# simply-cpp-ai

Wrappers for common AI tooling: ONNX, YOLO, and more.

The public API uses the `sc` namespace and builds on `simply-cpp` (sc-core) and `simply-cpp-image` (sc-image).

## Install

### Homebrew (macOS)

```bash
brew tap roelofrossouw/sc
brew install simply-cpp simply-cpp-image simply-cpp-ai
```

### apt (Ubuntu)

```bash
sudo curl -fsSL https://apt.roelof.co.za/setup.sh | bash
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
- **ONNX Runtime** - on macOS, installed automatically via brew if missing. On Linux there is currently no apt package for it: the build downloads a prebuilt CUDA build straight from ONNX Runtime's GitHub releases and installs it under `/usr/local` (needs root). If you're building from source on Linux without a matching NVIDIA/CUDA setup, install your own ONNX Runtime build first so `find_package(onnxruntime CONFIG REQUIRED)` succeeds before this step runs.
- **ONNX model files** - a separate `simply-cpp-models` package (same tap/repo as above) supplies the actual `.onnx` model weights. CMake auto-installs it at configure time if missing, so you generally don't need to do anything beyond having the Homebrew tap or apt repo set up.

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
