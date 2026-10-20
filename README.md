# simply-cpp-ai

Wrappers for common AI tooling: ONNX, YOLO, and more.

**Documentation:** the [simply-cpp wiki](https://github.com/roelofrossouw/simply-cpp/wiki) has an [AI guide](https://github.com/roelofrossouw/simply-cpp/wiki/AI) and the [sc-ai reference](https://github.com/roelofrossouw/simply-cpp/wiki/Reference-sc-ai), plus [getting started](https://github.com/roelofrossouw/simply-cpp/wiki/Getting-Started) for the whole suite.

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


## Model licences

The models in `simply-cpp-models` keep their upstream licences, not this
module's Apache-2.0; `/opt/simply-cpp/models/NOTICE.md` (apt) or the formula's
`share/simply-cpp-models/NOTICE.md` (Homebrew) lists them. In short: the
InsightFace face models (`det_10g`, `w600k_r50`) are for **non-commercial
research only**; the Ultralytics YOLO models (`yolo26n`, `yolo26m`, and
`numberplate`, trained on `yolo26n`) are **AGPL-3.0** unless you have an Ultralytics Enterprise
Licence; the PaddleOCR models are Apache-2.0. Check these before using sc-ai's
face or object detection in a product.

## Dependencies

- **simply-cpp (sc-core) and simply-cpp-image (sc-image)** - real dependencies of sc-ai's CMake package. Neither the Homebrew formula nor the apt package currently pulls them in automatically, so install all three explicitly (see above).
- **nlohmann_json** - `nlohmann-json3-dev` on apt, `nlohmann-json` on brew; installed automatically if missing when building from source.
- **ONNX Runtime** - `simply-cpp-onnxruntime` on apt (a repackaged build of ONNX Runtime's own GPU/CUDA release - a matching NVIDIA/CUDA setup needs to already be present), `onnxruntime` on brew; installed automatically if missing when building from source.
- **ONNX model files** - a separate `simply-cpp-models` package supplies the actual `.onnx` model weights; installed automatically if missing when building from source.

Both of the above come from the same Homebrew tap / apt repo as `simply-cpp` itself, and CMake registers that tap/repo automatically the first time it needs to - nothing to set up by hand first, even on a completely fresh machine.

## Usage

`sc-ai`'s own CMake package doesn't yet declare its dependency on `sc-image`, so when consuming an installed package, find that one explicitly too:

```cmake
find_package(sc-ai CONFIG REQUIRED)  # brings sc-image and sc-core

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

Two COCO object detectors come with simply-cpp-models: `yolo26n`, the nano model (quick: about
11 ms a frame on an Apple M-series CPU), and `yolo26m`, the medium one (about 2.5 times as long,
but much surer, and it finds what the nano model misses, such as people in close-up dashcam
footage). Both take a 640 x 640 picture and give up to 10 detections (they were exported with `max_det=10`), so they're used the same
way: `sc::yolo yolo{"yolo26m"};`.

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

With `--video` it does the same for a video (MP4, MOV, MKV, AVI and the rest, read a frame at a
time by sc-image's `sc::video`), a frame a second by default (`--every SECONDS`, 0 for every frame),
listing what's in each and then how many frames each kind of object was in. The model loads once,
and the frames between are skipped without being decoded. `--threshold PERCENT` (default 30) lists
less certain detections too, and `--model yolo26m` uses the medium model, slower (about 2.5 times
as long) but much surer, where the nano one (the default) misses things:

```bash
sc-ai-demo --video dashcam.mp4 --every 2 --threshold 20
```

```
  00:00.000  person, cell phone
  00:01.991  person
  ...
In 8 frames, in 00:00:00.097412000
  cell phone: in 3 frames
  person: in 8 frames
```

`--json` prints only JSON, for another program to read, and a video reads the same as a photo:
the detections as `yolo.to_json()` gives them (`[centre x %, centre y %, width %, height %,
confidence %, type]`), for a video each kind of object's surest one (`yolo.detect(clip, every)`),
most certain first:

```json
[[38.14, 56.46, 75.2, 84.97, 93.6, 0], [37.64, 73.66, 11.68, 7.25, 68.8, 67]]
```

With `--full` (`sc::yolo::detail::full`) a video gives a summary instead: for each kind of object,
how many frames it was in, how sure it was at most, and that frame's number, time and detections
(all of them, as for a photo):

```json
{"frames": 15, "every": 1000, "objects": {
  "person": {"count": 15, "confidence": 93.6, "frame": 45, "time": 2987,
             "detections": [[38.14, 56.46, 75.2, 84.97, 93.6, 0], [37.41, 72.97, 11.19, 13.02, 32.1, 67]]},
  "cell phone": {"count": 9, "confidence": 68.8, "frame": 105, "time": 6969, "detections": [...]}}}
```

Without a photo or video it uses a small rendered image with nothing in it, which still
shows the model, ONNX Runtime and sc-image loading and running. It is installed
with the runtime package (`simply-cpp-ai`), so it works without the `-dev`
package. It is a demonstration, not a test, so CTest doesn't run it.
`sc::yolo::object_name(type)` gives the name of a detection's type.

Its source is `examples/sc-ai-demo.cpp`; the code below is copied from it at
configure time, so it always matches code that compiles:

<!-- sc-example: examples/sc-ai-demo.cpp -->
```cpp
sc::console::heading("Loading the model");
const auto model = args.as<std::string>("model"); // a name in simply-cpp-models, or a file
sc::yolo yolo{model};
sc::console::show_text("sc::yolo yolo{\"" + model + "\"};", "loaded in " + std::string(sw));
yolo.set_threshold(threshold / 100); // a fraction: 0.3 lists what it's at least 30% sure of

if (video_file.empty()) {
    sc::console::heading("Detecting objects");
    sc::timer detecting;
    yolo.detect(std::filesystem::path{source});
    sc::console::show_text("yolo.detect(\"" + source + "\");", "done in " + std::string(detecting));

    // Each detection: centre x %, centre y %, width %, height %, confidence %, type.
    const auto detections = yolo.to_json();
    sc::console::heading(std::to_string(detections.size()) + (detections.size() == 1 ? " object found" : " objects found"));
    for (const auto &found: detections) sc::console::note(described(found));
} else {
    // A video: sc-image's sc::video reads it a frame at a time, each frame an sc::image.
    sc::console::heading("Detecting objects in a video");
    sc::video clip{video_file};
    std::ostringstream about;
    about << clip.size() << ", " << clip.fps() << " frames a second, " << clock(clip.duration());
    sc::console::show_text("sc::video clip{\"" + video_file.string() + "\"};", about.str());

    std::map<std::string, int> frames_with; // how many frames each kind of object was in
    int looked_at = 0;
    sc::timer detecting;
    while (const auto frame = clip.next(every)) { // the frames between are skipped, not decoded
        yolo.detect(frame->picture);
        ++looked_at;
        const auto detections = yolo.to_json();
        std::map<std::string, int> here;
        for (const auto &found: detections) ++here[sc::yolo::object_name(found[5].get<int>())];
        std::string line = clock(frame->time) + "  ";
        if (here.empty()) line += "nothing";
        for (const auto &[name, count]: here) {
            line += (line.ends_with("  ") ? "" : ", ") + (count > 1 ? std::to_string(count) + " x " : "") + name;
            ++frames_with[name];
        }
        sc::console::note(line);
    }
    sc::console::heading("In " + std::to_string(looked_at) + " frames, in " + std::string(detecting));
    if (frames_with.empty()) sc::console::note("nothing found");
    for (const auto &[name, count]: frames_with)
        sc::console::note(name + ": in " + std::to_string(count) + (count == 1 ? " frame" : " frames"));
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
