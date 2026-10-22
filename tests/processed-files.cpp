// process-id's choice of images and where it puts them once read, in a scratch folder: a recursive
// scan skipping the done folder (by name anywhere, and by path), and moves that keep subfolders and
// never overwrite.
#include "../apps/processed_files.h"

#include "sc_test.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {
    void touch(const fs::path &file) {
        fs::create_directories(file.parent_path());
        std::ofstream{file} << "x";
    }

    std::vector<std::string> names(const std::vector<fs::path> &files, const fs::path &base) {
        std::vector<std::string> out;
        for (const auto &file: files) out.push_back(fs::relative(file, base).generic_string());
        return out;
    }
}

int main() {
    using processed_files::done_folder;
    using processed_files::images_in;
    using processed_files::move_to_done;
    const auto root = fs::temp_directory_path() / ("process-id-test-" + std::to_string(std::random_device{}()));
    touch(root / "a.jpg");
    touch(root / "b.PNG");
    touch(root / "notes.txt");
    touch(root / "2026" / "c.jpeg");
    touch(root / "2026" / "done" / "old.jpg");  // a done folder deeper down: skipped too
    touch(root / "done" / "x.jpg");             // the done folder itself
    touch(root / "archive" / "y.jpg");

    SECTION("The done folder: under the input folder unless absolute");
    {
        CHECK(done_folder(root, "done") == root / "done");
        CHECK(done_folder(root, "done/") == root / "done");
        CHECK(done_folder(root / "a.jpg", "done") == root / "done");   // a file: beside it
        CHECK(done_folder(root, "/tmp/elsewhere") == fs::path{"/tmp/elsewhere"});
    }

    SECTION("Images in a folder: not its folders, unless recursive; never the done folders");
    {
        const auto done = done_folder(root, "done");
        CHECK(names(images_in(root, done, false), root) == (std::vector<std::string>{"a.jpg", "b.PNG"}));
        CHECK(names(images_in(root, done, true), root) == (std::vector<std::string>{"2026/c.jpeg", "a.jpg", "archive/y.jpg", "b.PNG"}));
        // Another name: folders called done are read again, and archive is the one skipped.
        CHECK(names(images_in(root, done_folder(root, "archive"), true), root) ==
              (std::vector<std::string>{"2026/c.jpeg", "2026/done/old.jpg", "a.jpg", "b.PNG", "done/x.jpg"}));
        CHECK(names(images_in(root / "a.jpg", done, true), root) == (std::vector<std::string>{"a.jpg"}));
        CHECK_THROWS_AS((void) images_in(root / "missing", done, false), std::runtime_error);
    }

    SECTION("Moving read files: subfolders kept, nothing overwritten");
    {
        const auto done = done_folder(root, "done");
        CHECK(move_to_done(root / "2026" / "c.jpeg", root, done) == done / "2026" / "c.jpeg");
        CHECK(!fs::exists(root / "2026" / "c.jpeg"));
        CHECK(fs::exists(done / "2026" / "c.jpeg"));
        touch(root / "x.jpg"); // done/x.jpg is already there
        CHECK(move_to_done(root / "x.jpg", root, done) == done / "x-2.jpg");
        touch(root / "x.jpg");
        CHECK(move_to_done(root / "x.jpg", root, done) == done / "x-3.jpg");
        CHECK(fs::exists(done / "x.jpg") && fs::exists(done / "x-2.jpg"));
        // A single file given: into the done folder beside it.
        CHECK(move_to_done(root / "a.jpg", root / "a.jpg", done_folder(root / "a.jpg", "done")) == done / "a.jpg");
        // A second run finds what's left, not what was moved.
        CHECK(names(images_in(root, done, true), root) == (std::vector<std::string>{"archive/y.jpg", "b.PNG"}));
    }

    fs::remove_all(root);
    TEST_SUMMARY();
}
