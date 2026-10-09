// South African identity documents with sc::identity: checking an identity number and what it
// encodes, and - given a photo of an ID card, ID book or passport - reading one with OCR. Each line
// shows a call, as written, and what it returned. The numbers below are made up.

#include <iostream>
#include <string>
#include <filesystem>

#include <sc.h>

#include <identity.h>

int main(int argc, char **argv) {
    try {
        sc::console::title("simply-cpp ai: South African identity documents");

        sc::console::heading("Checking an identity number");
        sc::console::note("YYMMDD birth date, four digits for sex (5000 and up is male), citizenship, and a Luhn check digit.");
        SC_SHOW(sc::identity::valid_id_number("8001015009087"));
        SC_SHOW(sc::identity::id_date_of_birth("8001015009087"));
        SC_SHOW(sc::identity::id_sex("8001015009087"));
        SC_SHOW(sc::identity::id_sex("9202204720083"));

        sc::console::subheading("Numbers that don't pass");
        SC_SHOW(sc::identity::valid_id_number("8001015009088")); // wrong check digit
        SC_SHOW(sc::identity::valid_id_number("8013015009082")); // a correct check digit, but no 13th month
        SC_SHOW(sc::identity::id_date_of_birth("12345"));        // empty when the number isn't valid

        sc::console::heading("Reading a document");
        if (argc < 2) {
            sc::console::note("Pass a photo of an ID card, ID book or passport to read one: sc-ai-identity <photo>");
            return 0;
        }
        sc::timer sw;
        const sc::identity document{std::filesystem::path{argv[1]}};
        sc::console::show_text("const sc::identity document{photo};", "read in " + std::string(sw));
        SC_SHOW(document.document_type_name());
        SC_SHOW(document.has_id_number());
        SC_SHOW(document.id_number());
        SC_SHOW(std::string(document.confidence())); // how many independent checks agreed
        SC_SHOW(document.checks().to_json().dump());
        sc::console::show_text("document.to_json().dump(2)", document.to_json().dump(2));
    } catch (const std::exception &error) {
        std::cerr << "sc-ai-identity: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
