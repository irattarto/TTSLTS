#!/usr/bin/env bash
# Downloads the prebuilt sherpa-onnx native libraries (libsherpa-onnx-jni.so and
# libonnxruntime.so) into app/src/main/jniLibs. Run once before building.
set -euo pipefail

SHERPA_ONNX_VERSION="${SHERPA_ONNX_VERSION:-v1.13.8}"
cd "$(dirname "$0")"

archive="sherpa-onnx-${SHERPA_ONNX_VERSION}-android.tar.bz2"
url="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_ONNX_VERSION}/${archive}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fSL --retry 3 -o "$tmp/$archive" "$url"
tar -xjf "$tmp/$archive" -C "$tmp"

rm -rf app/src/main/jniLibs
mkdir -p app/src/main/jniLibs
for abi in arm64-v8a armeabi-v7a x86_64; do
  cp -r "$tmp/jniLibs/$abi" "app/src/main/jniLibs/$abi"
done
ls -lh app/src/main/jniLibs/*
