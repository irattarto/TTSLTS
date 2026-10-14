#!/usr/bin/env bash
# Builds a .deb from a PyInstaller bundle.
# Usage: build-deb.sh BUNDLE_DIR ICON_PNG VERSION OUTPUT_DEB
set -euo pipefail

bundle="$(realpath "$1")"
icon="$(realpath "$2")"
version="$3"
output="$4"
here="$(cd "$(dirname "$0")" && pwd)"

arch="$(dpkg --print-architecture)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

install -d "$stage/opt/ttslts" "$stage/usr/bin" "$stage/usr/share/applications" \
  "$stage/usr/share/icons/hicolor/256x256/apps" "$stage/DEBIAN"
cp -a "$bundle/." "$stage/opt/ttslts/"
ln -s /opt/ttslts/TTSLTS "$stage/usr/bin/ttslts"
install -m 644 "$here/ttslts.desktop" "$stage/usr/share/applications/ttslts.desktop"
install -m 644 "$icon" "$stage/usr/share/icons/hicolor/256x256/apps/ttslts.png"

size_kb="$(du -sk "$stage" | cut -f1)"
cat > "$stage/DEBIAN/control" <<CONTROL
Package: ttslts
Version: $version
Section: sound
Priority: optional
Architecture: $arch
Installed-Size: $size_kb
Depends: libportaudio2, libegl1, libgl1, libxkbcommon-x11-0, libxcb-cursor0, libfontconfig1, libdbus-1-3
Maintainer: TTSLTS
Description: Offline text to speech and real-time voice changer
 System tray app with Hungarian text to speech (Piper) and an RVC v2 voice changer.
 Runs fully offline after the one-time model download.
CONTROL

dpkg-deb --root-owner-group -Zxz --build "$stage" "$output"
