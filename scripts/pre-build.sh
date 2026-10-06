#!/bin/bash
set -euo pipefail

model_version=$(grep Version pkg/DEBIAN/control | cut -d: -f2 | xargs)
model_file="../opt/simply-cpp-models_${model_version}_amd64.deb"

apt-get update

installed_version=$(dpkg-query -W -f='${Version}' simply-cpp-models 2>/dev/null || true)
candidate_version=$(apt-cache policy simply-cpp-models | awk '/Candidate:/ {print $2}')

if [ "$candidate_version" = "$model_version" ]; then
    echo "Using published simply-cpp-models $model_version"
    apt-get install -y simply-cpp-models
    exit 0
fi

if [ "$candidate_version" != "(none)" ] &&
   dpkg --compare-versions "$candidate_version" gt "$model_version"; then
    echo "Using newer published simply-cpp-models $candidate_version"
    apt-get install -y simply-cpp-models
    exit 0
fi

if [ "$installed_version" = "$model_version" ]; then
    echo "Using installed simply-cpp-models $model_version"
    exit 0
fi

if [ -n "$installed_version" ] &&
   dpkg --compare-versions "$installed_version" gt "$model_version"; then
    echo "Using newer installed simply-cpp-models $installed_version"
    exit 0
fi

echo "Building simply-cpp-models $model_version"
dpkg-deb --build pkg
mv pkg.deb "$model_file"

pushd /var/www/build/repo || exit
  export GNUPGHOME=/var/www/build/signing
  # remove from every dist before re-including anywhere: reprepro refuses to
  # include a same-version file with different content into one dist while
  # another still references the old pool copy under that same version+name.
  reprepro remove jammy simply-cpp-models
  reprepro remove noble simply-cpp-models
  reprepro remove resolute simply-cpp-models
  reprepro includedeb jammy "$model_file"
  reprepro includedeb noble "$model_file"
  reprepro includedeb resolute "$model_file"
popd || exit

apt-get update
apt-get install -y simply-cpp-models
