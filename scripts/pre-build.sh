model_version=$(grep Version pkg/DEBIAN/control | cut -d: -f2 | xargs)
model_file="../opt/simply-cpp-models_${model_version}_amd64.deb"

# Always rebuild: a file-exists check here meant an earlier incomplete/stale
# build under the same version string (e.g. a model added to pkg/ without a
# version bump, or an interrupted rsync) would silently persist forever,
# since nothing ever re-triggered a rebuild for that already-"created"
# version. dpkg-deb --build is cheap, so there's no real cost to redoing it.
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

# This server's own apt package lists were last refreshed before the includedeb
# calls above, so find_or_install_package(sc-models ...) below would otherwise
# see this server's already-installed (older) simply-cpp-models as "the newest
# version" and never notice a new one just got published.
apt-get update

# find_or_install_package() only installs when the package can't be found at
# all - an older-but-present simply-cpp-models satisfies that check and is
# never upgraded, so the build would keep using stale models forever once
# this server has installed it once. Upgrade it explicitly, here, where a
# new version just got published - this doesn't touch the general-purpose
# macro's "only act when genuinely missing" behaviour for every other
# dependency.
if dpkg -l simply-cpp-models > /dev/null 2>&1; then
    apt-get install -y --only-upgrade simply-cpp-models
fi
