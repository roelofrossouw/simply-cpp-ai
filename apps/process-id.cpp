// Reads identity documents (sc::identity: the document's text, its ID or passport number, names and
// faces) from an image, or every image in a folder, and stores each face's embedding in
// truckassist.id_embedding with the document's details.

#include <sc.h>

#include "identity.h"
#include <params.h>
#include <pgsql.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {
    const string query{
        R"(insert into truckassist.id_embedding
         (file, number, confidence, type, id_number, passport, names, surname, embedding)
         values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          returning id
)"
    };

    bool is_image(const filesystem::path &file) {
        string extension = file.extension().string();
        ranges::transform(extension, extension.begin(), [](const unsigned char c) { return static_cast<char>(tolower(c)); });
        return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".tif" || extension == ".tiff" ||
               extension == ".bmp" || extension == ".webp";
    }

    // The images to read: input itself when it's a file, else the images in the folder, by name.
    vector<filesystem::path> images_in(const filesystem::path &input) {
        if (filesystem::is_regular_file(input)) return {input};
        if (!filesystem::is_directory(input)) throw runtime_error{input.string() + " is neither an image nor a folder"};
        vector<filesystem::path> images;
        for (const auto &entry: filesystem::directory_iterator(input))
            if (entry.is_regular_file() && is_image(entry.path())) images.push_back(entry.path());
        ranges::sort(images);
        return images;
    }
}

int main(int argc, char **argv) {
    const sc::params args{argc, argv, {
        {"name", "process-id"},
        {"description", "Reads identity documents (the ID or passport number, names and faces) from an image, or every "
                        "image in a folder (JPEG, PNG, TIFF, BMP or WebP), and stores each face's embedding in "
                        "truckassist.id_embedding. The database settings left out come from libpq's environment: "
                        "PGHOST and PGPORT, PGDATABASE, PGUSER, and PGPASSWORD or ~/.pgpass."},
        {"options", {
            {"server", {{"short", "s"}, {"value", "HOST[:PORT]"},
                        {"help", "the database server, or several for failover: \"db1;db2:5433\" (default: PGHOST, PGPORT)"}}},
            {"database", {{"short", "d"}, {"value", "NAME"}, {"help", "the database (default: PGDATABASE)"}}},
            {"user", {{"short", "U"}, {"value", "NAME"}, {"help", "the database user (default: PGUSER)"}}},
            {"password", {{"value", "TEXT"},
                          {"help", "the user's password (default: PGPASSWORD or ~/.pgpass, which are safer: a password "
                                   "given here shows in the process list and the shell's history)"}}}}},
        {"positional", {
            {"input", {{"required", false}, {"help", "an image, or a folder of them (default: the current folder)"}}}}}}};

    try {
        const string input = args.as<string>("input", ".");
        const auto images = images_in(input);
        if (images.empty()) {
            cerr << "process-id: no images in " << input << '\n';
            return 1;
        }
        const sc::postgres db(args.as<string>("server", ""), args.as<string>("database", ""), args.as<string>("user", ""),
                              args.as<string>("password", ""));
        sc::timer sw;

        for (const auto &file: images) {
            sc::identity id(file, true);
            if (!id.has_id_number()) {
                cerr << "Could not find the ID " << file.filename() << endl;
                cerr << "Info: " << id.all_text() << endl;
                continue;
            }
            cout << sw
                    << ", " << id.confidence()
                    << ", " << id.id_number()
                    << ", " << id.passport_number()
                    << ", " << id.names()
                    << ", " << id.surname()
                    << ", " << id.faces().size() << " faces"
                    << " ==> " << file.filename()
                    << endl;
            sw.reset();
            int facenum = 0;
            for (const auto &face: id.faces()) {
                vector<string> parameters{
                    file.filename().string(),
                    to_string(++facenum),
                    to_string(static_cast<int>(id.confidence())),
                    id.document_type_name(),
                    id.id_number(),
                    id.passport_number(),
                    id.names(),
                    id.surname(),
                    face.to_json()["features"].dump()
                };
                auto db_result = db.exec(query, parameters);
                cout << sw << " DB " << db_result[0]["id"] << endl;
                sw.reset();
            }
        }
    } catch (const exception &error) {
        cerr << "process-id: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
