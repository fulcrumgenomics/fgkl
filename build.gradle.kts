plugins {
    `java-library`
}

group = "com.fulcrumgenomics"
version = "0.1.0-SNAPSHOT"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
    withJavadocJar()
    withSourcesJar()
}

repositories {
    mavenCentral()
}

dependencies {
    api("org.broadinstitute:gatk-native-bindings:1.1.0")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}

tasks.test {
    useJUnitPlatform()
    // Assertions and JNI checks catch marshalling mistakes in the native layer early.
    jvmArgs("-Xcheck:jni", "-ea")
}

tasks.javadoc {
    options.encoding = "UTF-8"
    (options as StandardJavadocDocletOptions).addBooleanOption("Xdoclint:none", true)
}

// ---------------------------------------------------------------------------
// Native build: cargo builds the JNI cdylib for the current (or FGKL_TARGET) platform and the
// result is copied under src/main/resources/native/<os>-<arch>/ for NativeLoader.
// ---------------------------------------------------------------------------

/** Canonical platform string such as "osx-aarch64" or "linux-x86_64". */
fun detectPlatform(): String {
    val osName = System.getProperty("os.name").lowercase()
    val os = when {
        osName.contains("mac") || osName.contains("darwin") -> "osx"
        osName.contains("linux") -> "linux"
        osName.contains("win") -> "windows"
        else -> error("Unsupported OS: $osName")
    }
    val archName = System.getProperty("os.arch").lowercase()
    val arch = when {
        archName == "amd64" || archName == "x86_64" -> "x86_64"
        archName == "aarch64" || archName == "arm64" -> "aarch64"
        else -> error("Unsupported architecture: $archName")
    }
    return "$os-$arch"
}

/** The Rust target triple for a platform string, used when cross-compiling. */
fun rustTarget(platform: String): String = when (platform) {
    "linux-x86_64" -> "x86_64-unknown-linux-gnu"
    "linux-aarch64" -> "aarch64-unknown-linux-gnu"
    "osx-x86_64" -> "x86_64-apple-darwin"
    "osx-aarch64" -> "aarch64-apple-darwin"
    "windows-x86_64" -> "x86_64-pc-windows-msvc"
    else -> error("No Rust target for $platform")
}

val hostPlatform = detectPlatform()
val platform = System.getenv("FGKL_PLATFORM") ?: hostPlatform
val crossCompiling = platform != hostPlatform
val libFileName = when {
    platform.startsWith("osx") -> "libfgkl.dylib"
    platform.startsWith("linux") -> "libfgkl.so"
    else -> "fgkl.dll"
}
val cargoOutputDir = if (crossCompiling) file("target/${rustTarget(platform)}/release") else file("target/release")
val nativeOutputDir = file("src/main/resources/native/$platform")

val buildNative by tasks.registering(Exec::class) {
    description = "Builds the fgkl JNI library with cargo and copies it into the resources tree."
    inputs.files(fileTree("crates") { exclude("**/target/**") }, "Cargo.toml", "Cargo.lock")
    outputs.file(nativeOutputDir.resolve(libFileName))

    // Skip when the library was provided some other way, e.g. downloaded from CI.
    onlyIf { !nativeOutputDir.resolve(libFileName).exists() || !gradle.startParameter.isOffline }

    val cargoArgs = mutableListOf("cargo", "build", "--release", "-p", "fgkl-jni")
    if (crossCompiling) cargoArgs.addAll(listOf("--target", rustTarget(platform)))
    commandLine(cargoArgs)

    doLast {
        nativeOutputDir.mkdirs()
        val built = cargoOutputDir.resolve(libFileName)
        require(built.exists()) { "cargo did not produce $built" }
        built.copyTo(nativeOutputDir.resolve(libFileName), overwrite = true)
    }
}

tasks.named("processResources") {
    dependsOn(buildNative)
}

tasks.named("sourcesJar") {
    dependsOn(buildNative)
}
