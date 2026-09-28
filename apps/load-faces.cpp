#include <facedetector.h>
#include <postgres.h>
#include <iostream>

using namespace std;

/*
-- select * from truckassist.facedata
-- create table truckassist.detection (id serial, embedding vector(512));
-- grant insert on table truckassist.detection to www;
-- grant all on sequence truckassist.detection_id_seq to www;
select * from truckassist.detection limit 10;
*/

int main(int argc, char **argv) {
    string query{"insert into truckassist.detection(embedding) values ($1)"};
    filesystem::path file{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/d1333 id.jpg"};
    sc::facedetector fd;
    sc::postgres dev("devdb", "1web", "www");
    auto faces = fd.detect(file);
    for (auto &face: faces) {
        auto feats_string = face.to_json()["features"].dump();
        auto r = dev.exec(query, {feats_string});
    }
    return 0;
}
