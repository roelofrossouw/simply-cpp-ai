#include <facedetector.h>
#include <sc_postgres.h>
#include <iostream>

using namespace std;

const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};

/*
-- select * from truckassist.facedata
-- create table truckassist.detection (id serial, embedding vector(512));
-- grant insert on table truckassist.detection to www;
-- grant all on sequence truckassist.detection_id_seq to www;
select * from truckassist.detection limit 10;
*/

int main(int argc, char **argv) {
    string query{"insert into truckassist.detection(file, number, embedding) values ($1, $2, $3)"};
    // const filesystem::path file{image_path / "d1333 id.jpg"};
    const sc::facedetector fd;
    const sc::postgres dev("devdb", "1web", "www");
    for (const auto &file: filesystem::directory_iterator(image_path)) {
        if (!file.is_regular_file()) continue;
        auto faces = fd.detect(file);
        cout << "Running file " << file.path().filename() << " (" << faces.size() << " faces)\n";
        int imgno = 1;
        for (auto &face: faces) {
            const auto feats_string = face.to_json()["features"].dump();
            const vector parameters{file.path().filename().string(), to_string(imgno++), feats_string};
            auto r = dev.exec(query, parameters);
        }
    }
    return 0;
}
