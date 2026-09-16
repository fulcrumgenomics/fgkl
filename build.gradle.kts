plugins {
    `java-library`
}

group = "com.fulcrumgenomics"
version = "0.1.0-SNAPSHOT"

java {
    // Compile against the JDK 17 API, not merely at source level 17, so the JAR cannot pick up
    // newer APIs that GATK's JDK 17 lacks.
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
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

tasks.jar {
    manifest {
        attributes(
            "Automatic-Module-Name" to "com.fulcrumgenomics.fgkl",
            "Implementation-Title" to project.name,
            "Implementation-Version" to project.version,
        )
    }
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
// Native build: cargo builds the JNI cdylib for the host (or FGKL_PLATFORM) platform into
// build/native/<os>-<arch>/ and processResources packages it at native/<os>-<arch>/ in the JAR,
// where NativeLoader looks for it. A multi-platform JAR is assembled by dropping the other
// platforms' libraries into build/native/ before packaging.
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
val rustTargetTriple = rustTarget(platform) // validates the platform string up front
val crossCompiling = platform != hostPlatform
val libFileName = when {
    platform.startsWith("osx") -> "libfgkl.dylib"
    platform.startsWith("linux") -> "libfgkl.so"
    else -> "fgkl.dll"
}
val cargoOutputDir = if (crossCompiling) file("target/$rustTargetTriple/release") else file("target/release")
val nativeDir = layout.buildDirectory.dir("native")

val buildNative by tasks.registering(Exec::class) {
    description = "Builds the fgkl JNI library with cargo into build/native/<platform>/."
    inputs.files(fileTree("crates") { exclude("**/target/**") }, "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml")
    inputs.property("platform", platform)
    outputs.file(nativeDir.map { it.file("$platform/$libFileName") })

    val cargoArgs = mutableListOf("cargo", "build", "--release", "--locked", "-p", "fgkl-jni")
    if (crossCompiling) cargoArgs.addAll(listOf("--target", rustTargetTriple))
    commandLine(cargoArgs)

    doLast {
        val built = cargoOutputDir.resolve(libFileName)
        require(built.exists()) { "cargo did not produce $built" }
        val out = nativeDir.get().dir(platform).asFile
        out.mkdirs()
        built.copyTo(out.resolve(libFileName), overwrite = true)
    }
}

tasks.processResources {
    dependsOn(buildNative)
    from(nativeDir) { into("native") }
}
