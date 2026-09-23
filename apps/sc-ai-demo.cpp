#include <yolo.h>
#include <nlohmann/json.hpp>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Warning: no image filename was passed as a parameter." << endl;
        return 0;
    }

    sc::yolo yolo("yolo26n.onnx");
    yolo.detect(string(argv[1]));
    const auto result = static_cast<nlohmann::ordered_json>(yolo);
    cout << result.dump() << endl;
    return 0;
}
