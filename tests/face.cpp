#include "face.h"

#include <sc_test.h>
#include <chrono>
#include <filesystem>
#include <limits>
#include <stdexcept>

using namespace std;

int main() {
    SECTION("Feature-only face");
    {
        sc::face::feature_vector features{};
        features[0] = 3.0f;
        features[1] = 4.0f;
        const sc::face detected{features};

        CHECK(!detected.has_image());
        CHECK(detected.face_image() == nullptr);
        CHECK_NEAR(detected.features()[0], 0.6f, 1e-6f);
        CHECK_NEAR(detected.features()[1], 0.8f, 1e-6f);
        CHECK_NEAR(static_cast<double>(detected.similarity(detected)), 100.0, 1e-6);
    }

    SECTION("JSON and file persistence");
    {
        sc::face::feature_vector features{};
        features[0] = 1.0f;
        const sc::face original{features};
        const auto restored = sc::face::from_json(original.to_json());
        CHECK(!restored.has_image());
        CHECK_NEAR(restored.similarity(original), 100.0f, 1e-6f);

        const auto path = filesystem::temp_directory_path()
                          / ("sc-ai-face-features-" +
                             to_string(chrono::steady_clock::now().time_since_epoch().count()) + ".json");
        original.save(path);
        const auto loaded = sc::face::load(path);
        CHECK(!loaded.has_image());
        CHECK_NEAR(loaded.similarity(original), 100.0f, 1e-6f);
        filesystem::remove(path);
    }

    SECTION("Optional image");
    {
        const sc::image image{"resource/samples/card.jpg"}; // any image will do
        sc::face::feature_vector features{};
        features[0] = 1.0f;
        const sc::face with_image{features, image};
        CHECK(with_image.has_image());
        CHECK(with_image.face_image() != nullptr);
        CHECK_EQ(with_image.face_image()->size(), image.size());
        CHECK(!sc::face::from_json(with_image.to_json()).has_image());
    }

    SECTION("Invalid features");
    {
        sc::face::feature_vector zero_features{};
        CHECK_THROWS_AS(sc::face{zero_features}, invalid_argument);

        sc::face::feature_vector invalid_features{};
        invalid_features[0] = numeric_limits<float>::infinity();
        CHECK_THROWS_AS(sc::face{invalid_features}, invalid_argument);

        auto invalid_json = nlohmann::ordered_json{{"version", 1}, {"features", {1.0f}}};
        CHECK_THROWS_AS(sc::face::from_json(invalid_json), invalid_argument);
    }

    TEST_SUMMARY();
}
