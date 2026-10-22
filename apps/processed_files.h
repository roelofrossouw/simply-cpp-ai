#pragma once

#include <core.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Which images process-id reads, and where it puts them once read.
namespace processed_files {
    namespace fs = std::filesystem;

    inline bool is_image(const fs::path &file) {
        static const std::set<std::string> extensions{".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"};
        return extensions.contains(sc::lowercase(file.extension().string()));
    }

    // The folder read files go to: done as given when absolute, else under the input's folder (the
    // input itself, or a file's own folder).
    inline fs::path done_folder(const fs::path &input, const fs::path &done) {
        auto folder = done.is_absolute() ? done : [&] {
            const auto base = fs::is_directory(input) ? input : input.parent_path();
            return (base.empty() ? fs::path{"."} : base) / done;
        }();
        folder = folder.lexically_normal();
        if (folder.filename().empty()) folder = folder.parent_path(); // "done/"
        return folder;
    }

    // The images to read, by path: input itself when it's a file, else the images in the folder
    // (recursive: in its folders too). A recursive scan leaves out every folder named like the done
    // folder, and the done folder itself wherever it is, so nothing is read twice.
    inline std::vector<fs::path> images_in(const fs::path &input, const fs::path &done, const bool recursive) {
        if (fs::is_regular_file(input)) return {input};
        if (!fs::is_directory(input)) throw std::runtime_error{input.string() + " is neither an image nor a folder"};
        const auto done_name = done.filename();
        std::error_code ignored;
        const auto done_path = fs::weakly_canonical(done, ignored);
        const auto skipped = [&](const fs::path &folder) {
            return folder.filename() == done_name || (!done_path.empty() && fs::weakly_canonical(folder, ignored) == done_path);
        };
        std::vector<fs::path> images;
        if (!recursive) {
            for (const auto &entry: fs::directory_iterator(input))
                if (entry.is_regular_file() && is_image(entry.path())) images.push_back(entry.path());
        } else {
            for (auto it = fs::recursive_directory_iterator(input, fs::directory_options::skip_permission_denied);
                 it != fs::recursive_directory_iterator(); ++it) {
                if (it->is_directory() && skipped(it->path())) {
                    it.disable_recursion_pending();
                    continue;
                }
                if (it->is_regular_file() && is_image(it->path())) images.push_back(it->path());
            }
        }
        std::ranges::sort(images);
        return images;
    }

    // Moves a read file into done, keeping its place under the input folder (a recursive scan's
    // subfolders), and returns where it went. A file of that name already there isn't overwritten:
    // the new one gets "-2", "-3" ... before its extension.
    inline fs::path move_to_done(const fs::path &file, const fs::path &input, const fs::path &done) {
        const auto base = fs::is_directory(input) ? input : file.parent_path();
        const auto relative = fs::relative(file, base);
        auto target = done / relative;
        fs::create_directories(target.parent_path());
        for (int n = 2; fs::exists(target); ++n)
            target = done / relative.parent_path() / (relative.stem().string() + "-" + std::to_string(n) + relative.extension().string());
        std::error_code error;
        fs::rename(file, target, error);
        if (error) { // another file system: copy, then remove
            fs::copy_file(file, target);
            fs::remove(file);
        }
        return target;
    }
}
