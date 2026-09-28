#include "onnx.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

using namespace std;


const filesystem::path image_path{"/Users/roelof/simply-cpp-suite/modules/sc-ai/tests/resource/test/"};

int main() {
    sc::onnx detect("paddle_det_s");
    sc::onnx rec("paddle_rec_s");

    std::vector<std::string> dict;
    dict.emplace_back(" ");
    std::ifstream in("/opt/homebrew/share/simply-cpp-models/paddle_rec_s.txt");
    std::string line;
    while (std::getline(in, line)) dict.push_back(line);

    for (const auto &file: filesystem::directory_iterator(image_path)) {
        sc::timer sw;
        if (!file.is_regular_file()) continue;
        sc::image input{file.path().string()};
        input.snap_to_size(64, {input.size()});
        auto results = detect.process_image(input);
        cout << sw << " Detect " << endl;
        cout << file << endl;
        string words;
        for (const auto &output: results) {
            cv::Mat heatmap((int) output.shape[2], (int) output.shape[3], CV_32F, output.data);
            double minv, maxv;
            cv::minMaxLoc(heatmap, &minv, &maxv);

            cv::Mat mask;
            cv::threshold(heatmap, mask, 0.5f, 255, cv::THRESH_BINARY);
            mask.convertTo(mask, CV_8U);

            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            cout << sw << " Processed " << endl;

            for (const auto &contour: contours) {
                auto r = cv::boundingRect(contour);
                if (r.area() < 100) continue;
                auto block = input.crop(sc::rect_i{r.x - 5, r.y - 5, r.width + 10, r.height + 10});
                if (block.size().width() < block.size().height()) block.rotate(90);
                block.resize_to({0, 48});
                auto recs = rec.process_image(block)[0];
                auto timesteps = recs.shape[1];
                auto classes = recs.shape[2];
                int maxclasses = 1000;
                auto p = recs.data;

                string symbols;
                int last_class{};
                for (int t = 0; t < timesteps; ++t) {
                    int best_class = 0;
                    float best_score = 0.95;
                    for (int c = 0; c < maxclasses; ++c) {
                        float score = p[t * classes + c];
                        if (score > best_score) {
                            best_score = score;
                            best_class = c;
                        }
                    }
                    if (best_class == last_class) continue;
                    last_class = best_class;
                    if (best_class) symbols += dict[best_class];
                }
                words += symbols + '\n';
            }
        }
        cout << sw << " Recog " << endl;
        cout << words << endl;
        cout << sw << endl;
        if (!input.show(0)) break;
    }
    return 0;
}
