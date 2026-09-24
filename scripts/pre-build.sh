model_version=$(grep Version pkg/DEBIAN/control | cut -d: -f2 | xargs)
model_file="../opt/simply-cpp-models_${model_version}_amd64.deb"
if [ -f $model_file ]; then
    echo "Model package $model_version already created"
else
    echo "Creating a repo for onnx models used"
    dpkg-deb --build pkg
    mv pkg.deb $model_file
fi

pushd /var/www/build/repo || exit
  export GNUPGHOME=/var/www/build/signing
  reprepro list jammy | grep -q "simply-cpp-models.*${model_version}" || reprepro includedeb jammy "$model_file"
  reprepro list noble | grep -q "simply-cpp-models.*${model_version}" || reprepro includedeb noble "$model_file"
  reprepro list resolute | grep -q "simply-cpp-models.*${model_version}" || reprepro includedeb resolute "$model_file"
popd || exit

# This server's own apt package lists were last refreshed before the includedeb
# calls above, so find_or_install_package(sc-models ...) below would otherwise
# see this server's already-installed (older) simply-cpp-models as "the newest
# version" and never notice a new one just got published.
apt-get update
