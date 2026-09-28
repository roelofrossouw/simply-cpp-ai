#pragma once

#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "face.h"
#include <percent.h>
#include <nlohmann/json.hpp>

namespace sc {
    namespace impl {
        class identity_reader;
    }

    /// A South African identity document - smart ID card, green ID book or
    /// passport - read from a photograph.
    ///
    /// OCR text from the image is parsed for an identity number. Every number
    /// reported has at least passed a Luhn check; confidence() reports how many
    /// independent checks agreed.
    ///
    /// A passport is read from its machine readable zone, where the identity
    /// number sits under a check digit, so those reads verify themselves. A
    /// card or book is read from the printed number and corroborated against
    /// the printed birth date and sex.
    ///
    ///     const sc::identity document{"id.jpg"};
    ///     if (document.confidence() >= 90) store(document.to_json());
    ///     else queue_for_review(document);
    ///
    /// Reading is thread safe; each thread builds its own OCR models on first use.
    class identity {
    public:
        enum class document { unknown, id_card, id_book, passport };

        /// Independent corroborations of a recognised identity number. Each
        /// comes from a different part of the document, so agreement between
        /// them is meaningful rather than circular.
        struct verification {
            bool luhn{}; ///< the thirteenth digit is a correct Luhn check digit
            bool date_of_birth{}; ///< digits one to six match the printed date of birth
            bool sex{}; ///< digit seven matches the printed sex
            bool mrz{}; ///< read from a passport MRZ whose check digits agree

            /// Confidence score out of 100: 60 for the Luhn check, plus 25 for
            /// the date of birth, 10 for the sex and 35 for the MRZ.
            [[nodiscard]] int score() const noexcept;

            [[nodiscard]] int passed() const noexcept;

            [[nodiscard]] nlohmann::ordered_json to_json() const;
        };

        /// An empty result; document_type() is unknown and no fields are set.
        identity() = default;

        /// Reads the document in an image. Never throws for an unreadable
        /// document - check has_id_number() - but does throw when the file
        /// cannot be opened or the OCR models are unavailable.
        /// Set face_detection to also detect embeddings in the source image.
        explicit identity(const std::filesystem::path &image_path, bool face_detection = false);

        /// Reads a document, discarding any previous result, and returns
        /// whether an identity number was recognised. Face detection is
        /// optional and disabled by default.
        bool read(const std::filesystem::path &image_path, bool face_detection = false);

        [[nodiscard]] document document_type() const noexcept;
        [[nodiscard]] std::string document_type_name() const;

        [[nodiscard]] bool has_id_number() const noexcept;
        [[nodiscard]] const std::string &id_number() const noexcept;

        /// How much the recognised number can be trusted; zero when nothing was
        /// found. Reads above 90% agreed on at least three independent checks.
        [[nodiscard]] percent confidence() const;

        /// Which individual checks agreed.
        [[nodiscard]] const verification &checks() const noexcept;

        /// Mean PaddleOCR character confidence for the line containing the
        /// number. Unlike confidence(), this says nothing about correctness.
        [[nodiscard]] percent ocr_confidence() const;

        /// Face embeddings detected in the source image when face detection
        /// was requested. Images are not retained with the embeddings.
        [[nodiscard]] const std::vector<face> &faces() const noexcept;

        /// Rotation in degrees applied before OCR. Currently always zero.
        [[nodiscard]] int rotation() const noexcept;

        [[nodiscard]] const std::string &surname() const noexcept;
        [[nodiscard]] const std::string &names() const noexcept;
        [[nodiscard]] const std::string &date_of_birth() const noexcept; ///< ISO yyyy-mm-dd
        [[nodiscard]] const std::string &sex() const noexcept; ///< "M" or "F"
        [[nodiscard]] const std::string &nationality() const noexcept;
        [[nodiscard]] const std::string &country_of_birth() const noexcept;
        [[nodiscard]] const std::string &status() const noexcept;
        [[nodiscard]] const std::string &passport_number() const noexcept;
        [[nodiscard]] const std::string &date_of_issue() const noexcept; ///< ISO yyyy-mm-dd
        [[nodiscard]] const std::string &date_of_expiry() const noexcept; ///< ISO yyyy-mm-dd

        /// Citizenship encoded in the identity number: true for a citizen,
        /// false for a permanent resident, empty when no number was read.
        [[nodiscard]] std::optional<bool> citizen() const;

        [[nodiscard]] nlohmann::ordered_json to_json() const;
        operator nlohmann::ordered_json() const;
        operator std::string() const;
        friend std::ostream &operator<<(std::ostream &lhs, const identity &rhs);

        /// True when a string is thirteen digits forming a structurally valid
        /// South African identity number: plausible birth date, a citizenship
        /// digit of 0 or 1, and a correct Luhn check digit.
        [[nodiscard]] static bool valid_id_number(const std::string &number);

        /// The ISO birth date encoded in an identity number, or an empty string
        /// when the number is not valid. Two-digit years are resolved against
        /// the current year, so a year after it is treated as last century.
        [[nodiscard]] static std::string id_date_of_birth(const std::string &number);

        /// "M" or "F" as encoded in an identity number, or an empty string when
        /// the number is not valid.
        [[nodiscard]] static std::string id_sex(const std::string &number);

    private:
        document document_{document::unknown};
        std::string id_number_;
        verification checks_;
        double ocr_confidence_{};
        int rotation_{};
        std::string surname_;
        std::string names_;
        std::string date_of_birth_;
        std::string sex_;
        std::string nationality_;
        std::string country_of_birth_;
        std::string status_;
        std::string passport_number_;
        std::string date_of_issue_;
        std::string date_of_expiry_;
        std::vector<face> faces_;
        bool face_detection_performed_{};

        friend class impl::identity_reader;
    };
} // namespace sc
