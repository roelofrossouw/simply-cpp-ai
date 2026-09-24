# simply-cpp-models is a plain data package (a handful of .onnx files), not a real
# CMake-built one - this file exists purely so a consumer can find it the same way
# as every other dependency, via find_or_install_package(sc-models ...), instead of
# needing its own bespoke find_file()/apt/brew-install logic.
#
# The apt layout is fixed (this package always installs to /opt/simply-cpp/models),
# so the path is hardcoded rather than computed relative to this file's own install
# location the way the Homebrew side has to be.
set(SC_MODELS_DIR "/opt/simply-cpp/models")
set(sc-models_FOUND TRUE)
