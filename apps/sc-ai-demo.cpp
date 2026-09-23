#include <yolo.h>
#include <nlohmann/json.hpp>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Please specify an image to run detection on." << endl;
        return 0;
    }

    sc::yolo yolo("yolo26n.onnx");
    yolo.detect(string(argv[1]));
    cout << yolo.to_json().dump() << endl;
    return 0;
}
