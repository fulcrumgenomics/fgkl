#!/usr/bin/env bash
# Builds the fgkl native library for every supported platform, assembles and tests the JAR, and
# publishes it to Maven Central: a release when HEAD carries a version tag, a snapshot otherwise.
#
# Runs on a macOS aarch64 machine with: the pinned Rust toolchain plus the targets below, zig and
# cargo-zigbuild (Linux cross-compilation), Docker (Linux test runs), JDK 17, and credentials as
# environment variables (SONATYPE_USER, SONATYPE_PASS and, for a release, PGP_SECRET and
# PGP_PASSPHRASE) or as the equivalent Gradle properties in ~/.gradle/gradle.properties.
#
# Usage: publish.sh [--dry-run]     (--dry-run does everything except the upload)
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Platform name inside the JAR -> Rust target. The Linux libraries are linked against an old
# glibc (see scripts/test-linux.sh) so they load on old cluster distributions.
declare -A TARGETS=(
  [linux-x86_64]="x86_64-unknown-linux-gnu"
  [linux-aarch64]="aarch64-unknown-linux-gnu"
  [osx-x86_64]="x86_64-apple-darwin"
  [osx-aarch64]="aarch64-apple-darwin"
  [windows-x86_64]="x86_64-pc-windows-gnu"
)

for tool in cargo cargo-zigbuild zig docker java; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
[[ -z "$(git status --porcelain)" ]] || { echo "working tree is not clean" >&2; exit 1; }

version=$(./gradlew -q printVersion)
echo "publishing version $version"
if [[ "$version" != *-SNAPSHOT ]]; then
  cargo_version=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')
  [[ "$version" == "$cargo_version" ]] || { echo "tag version $version != Cargo version $cargo_version" >&2; exit 1; }
  if [[ -z "${PGP_SECRET:-}" ]] && ! grep -qs '^signingKey=' "$HOME/.gradle/gradle.properties"; then
    echo "a release needs PGP_SECRET in the environment or signingKey in ~/.gradle/gradle.properties" >&2; exit 1
  fi
fi

rm -rf build/native
for platform in "${!TARGETS[@]}"; do
  target=${TARGETS[$platform]}
  echo "== building $platform ($target)"
  case "$target" in
    *-linux-gnu) cargo zigbuild --release --locked -p fgkl-jni --target "$target.${GLIBC:-2.17}" ;;
    *-windows-*) cargo zigbuild --release --locked -p fgkl-jni --target "$target" ;;
    *)           cargo build --release --locked -p fgkl-jni --target "$target" ;;
  esac
  case "$platform" in
    linux-*)   lib=libfgkl.so ;;
    osx-*)     lib=libfgkl.dylib ;;
    windows-*) lib=fgkl.dll ;;
  esac
  mkdir -p "build/native/$platform"
  cp "target/$target/release/$lib" "build/native/$platform/$lib"
done
ls -la build/native/*/

# The host build would overwrite build/native/osx-aarch64 with the same bytes; skip it and
# package what was just built.
./gradlew build -x buildNative
jar=$(ls build/libs/fgkl-*.jar | grep -vE 'sources|javadoc')
unzip -l "$jar" | grep native/ || { echo "no native libraries in $jar" >&2; exit 1; }
if unzip -l build/libs/fgkl-*-sources.jar | grep -q native/; then
  echo "sources JAR contains a native library" >&2; exit 1
fi

# Test the Linux libraries in Docker (rebuilds them into the same build/native paths). The macOS
# aarch64 library was tested by ./gradlew build on this machine; the Windows library is untested
# here (the manual `windows` GitHub workflow covers it).
scripts/test-linux.sh

if (( DRY_RUN )); then
  echo "dry run: skipping upload of $version"
  exit 0
fi
if [[ "$version" == *-SNAPSHOT ]]; then
  ./gradlew -x buildNative publishToSonatype
else
  ./gradlew -x buildNative publishToSonatype closeAndReleaseSonatypeStagingRepository
fi
echo "published $version"
