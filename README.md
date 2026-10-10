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

## Dependencies

- **simply-cpp (sc-core) and simply-cpp-image (sc-image)** - real dependencies of sc-ai's CMake package. Neither the Homebrew formula nor the apt package currently pulls them in automatically, so install all three explicitly (see above).
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

sc::ocr reader; // paddle_det_monkt_5 and paddle_rec_monkt_latin from simply-cpp-models
reader.detect("document.jpg");
std::cout << reader.text() << std::endl;
std::cout << reader.to_json().dump(2) << std::endl;
```

Use the optional constructor arguments to select detection and recognition models, and an optional third path for a custom recognition dictionary. `reader.lines()` returns the recognised line records directly; `reader.display()` shows the source image with text boxes.

How it reads a page:

- **Upright first.** Since 1.6.0, `detect()` turns the image upright with PaddleOCR's document orientation classifier (`paddle_rotate`) before reading, so a photo taken sideways or upside down reads as text rather than noise. Boxes are still in the given image's coordinates; `reader.rotation()` says how far it was turned, `sc::ocr::upright_rotation(image)` gives the same answer on its own, and `reader.set_auto_rotate(false)` reads the image as given. Results on such photos differ from earlier versions.
- **Detection** scales the image to at most 1280 pixels on the longer side, looks again at a better scale when the text is very large or very small (small text in a large image is read in overlapping tiles), grows each region back to cover the whole line, and turns short regions (a lone letter) to the page's text direction.
- **Recognition** reads every line the same way up as most of the page, using a text line orientation classifier (`paddle_line_rotate`); a line is also read the other way only when that is clearly better.

Without the two classifier models (an older simply-cpp-models) it falls back to reading every line both ways.

`sc::identity` uses the same engine and models: it turns the photo upright, reads it, and parses the lines for document fields, trying the photo turned a quarter either way only when fields are missing. Tesseract is no longer required; identity recognition remains English-only.

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

`sc-ai-demo` runs YOLO object detection on a photo and lists what it found:

```bash
sc-ai-demo street.jpg
```

```
5 objects found
  person         92.5% sure, centred at (53.2%, 44.3%), 20.3% x 51.9% of the image
  surfboard      70.7% sure, centred at (50.8%, 72.6%), 16.3% x 17.5% of the image
  ...
```

Without a photo it uses a small rendered image with nothing in it, which still
shows the model, ONNX Runtime and sc-image loading and running. It is installed
with the runtime package (`simply-cpp-ai`), so it works without the `-dev`
package. It is a demonstration, not a test, so CTest doesn't run it.
`sc::yolo::object_name(type)` gives the name of a detection's type.

Its source is `examples/sc-ai-demo.cpp`; the code below is copied from it at
configure time, so it always matches code that compiles:

<!-- sc-example: examples/sc-ai-demo.cpp -->
```cpp
sc::console::heading("Loading the model");
sc::yolo yolo{"yolo26n.onnx"};
sc::console::show_text("sc::yolo yolo{\"yolo26n.onnx\"};", "loaded in " + std::string(sw));

sc::console::heading("Detecting objects");
sc::timer detecting;
yolo.detect(std::filesystem::path{source});
sc::console::show_text("yolo.detect(\"" + source + "\");", "done in " + std::string(detecting));

// Each detection: centre x %, centre y %, width %, height %, confidence %, type.
const auto detections = yolo.to_json();
sc::console::heading(std::to_string(detections.size()) + (detections.size() == 1 ? " object found" : " objects found"));
for (const auto &found: detections) {
    std::ostringstream line;
    line << std::left << std::setw(14) << sc::yolo::object_name(found[5].get<int>()) << std::right
         << std::fixed << std::setprecision(1) << std::setw(5) << found[4].get<double>()
         << "% sure, centred at (" << found[0].get<double>() << "%, " << found[1].get<double>() << "%), "
         << found[2].get<double>() << "% x " << found[3].get<double>() << "% of the image";
    sc::console::note(line.str());
}
```
<!-- /sc-example -->

More demos, installed alongside it:

| Demo | Shows |
|---|---|
| `sc-ai-ocr [image]` | reading text with `sc::ocr`: each line with its confidence and position; without an image it renders a few lines of text to read |
| `sc-ai-identity [photo]` | checking South African identity numbers (`valid_id_number`, `id_date_of_birth`, `id_sex`), and reading a photographed ID card, ID book or passport when given one |

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

The identity and OCR tests read specimen documents with made-up data (`tests/resource/samples`).
Real identity documents are never committed: put them in `tests/resource/private/documents`
(git-ignored), where face detection and a few extra checks look for them; without them those are
skipped.

## Other ways to use it

The packages above are the simplest route. sc-ai can also be built from source, with
CMake's `FetchContent` or as a git submodule (`add_subdirectory`), from
https://github.com/roelofrossouw/simply-cpp-ai. It builds sc-core and sc-image too when they aren't installed.
