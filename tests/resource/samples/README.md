# Specimen documents

Made-up identity documents for the tests and demos: a smart ID card (also turned sideways), a
passport data page with a machine readable zone whose check digits agree, and a green ID book page.
The people, numbers and dates are invented, and each is marked "SPECIMEN - NOT A VALID DOCUMENT".

The JPEGs are rendered from the SVGs beside them with `sc::svg2png` and `sc::image` (Arial or
Helvetica), so the tests read the same pixels on every machine. To change one, edit its SVG and
render it again:

```cpp
std::ofstream{"card.png", std::ios::binary} << sc::svg2png::FromFile("card.svg");
sc::image{"card.png"}.save("card.jpg");
```

Real documents never go in the repository: put them in `tests/resource/private/` (git-ignored),
where the tests that need them - real faces, real photographs - look for them, and skip without.
