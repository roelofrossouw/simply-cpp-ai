#include <sc.h>

#include "identity.h"
#include <postgres.h>

#include "ocr.h"

using namespace std;

const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};
const string query{
    R"(insert into truckassist.id_embedding
         (file, number, confidence, type, id_number, passport, names, surname, embedding)
         values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          returning id
)"
};

int main() {
    const sc::postgres dev("devdb", "1web", "www");
    sc::timer sw;

    for (const auto &file: filesystem::directory_iterator(image_path)) {
        // if (file.path().filename() != "ID-38b.jpg") continue;
        if (file.path().filename() != "ID-38.jpg") continue;
        // Could not find the ID "ID-10.jpg"")
        if (!file.is_regular_file()) continue;
        sc::identity id(file, true);
        if (!id.has_id_number()) {
            cerr << "Could not find the ID " << file.path().filename() << endl;
            continue;
        }
        cout << sw
                << " " << id.confidence()
                << " " << id.id_number()
                << " " << id.passport_number()
                << " " << id.names()
                << " " << id.surname()
                << " " << id.faces().size() << " faces"
                << " ==> " << file.path().filename()
                << endl;
        sw.reset();
        int facenum = 0;
        for (const auto &face: id.faces()) {
            vector<string> parameters{
                file.path().filename().string(),
                to_string(++facenum),
                to_string(static_cast<int>(id.confidence())),
                id.document_type_name(),
                id.id_number(),
                id.passport_number(),
                id.names(),
                id.surname(),
                face.to_json()["features"].dump()
            };
            // auto db_result = dev.exec(query, parameters);
            // cout << sw << " DB " << db_result[0]["id"] << endl;
            sw.reset();
        }
    }
    return 0;
}
