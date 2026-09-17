#!/usr/bin/env bash
# Cross-compiles the JNI library for Linux and runs the Java tests against it in Docker, one
# container per architecture. Needs zig, cargo-zigbuild and Docker; see CONTRIBUTING.md.
# Usage: scripts/test-linux.sh [amd64|arm64 ...]   (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."

# The Linux libraries link against this glibc so they load on old cluster distributions.
GLIBC=${GLIBC:-2.17}
ARCHES=("$@"); (( ${#ARCHES[@]} )) || ARCHES=(amd64 arm64)

for arch in "${ARCHES[@]}"; do
  case "$arch" in
    amd64) target=x86_64-unknown-linux-gnu; platform=linux-x86_64 ;;
    arm64) target=aarch64-unknown-linux-gnu; platform=linux-aarch64 ;;
    *) echo "unknown architecture $arch (amd64 or arm64)" >&2; exit 1 ;;
  esac
  echo "== building $platform"
  cargo zigbuild --release --locked -p fgkl-jni --target "$target.$GLIBC"
  mkdir -p "build/native/$platform"
  cp "target/$target/release/libfgkl.so" "build/native/$platform/"
done

# Gradle's resources tree mirrors the JAR's native/ layout and NativeLoader picks the platform
# at runtime. The Gradle cache is shared with the host so nothing is re-downloaded; file
# watching is off because the mounted volume does not support it.
for arch in "${ARCHES[@]}"; do
  echo "== testing on linux/$arch"
  docker run --rm --platform "linux/$arch" -v "$PWD:/work" -w /work -v "$HOME/.gradle:/root/.gradle" \
    eclipse-temurin:17-jdk bash -c "apt-get install -y -qq git >/dev/null 2>&1; \
      ./gradlew --no-daemon --console=plain -Dorg.gradle.vfs.watch=false test -x buildNative --rerun-tasks 2>&1 \
      | grep -E 'Task :test|BUILD|FAILED|Exception'; test \${PIPESTATUS[0]} -eq 0"
done
echo "linux tests passed: ${ARCHES[*]}"
