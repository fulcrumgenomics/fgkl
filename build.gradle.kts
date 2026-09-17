plugins {
    `java-library`
    `maven-publish`
    signing
    id("pl.allegro.tech.build.axion-release") version "1.21.3"
    id("io.github.gradle-nexus.publish-plugin") version "2.0.0"
}

group = "com.fulcrumgenomics"

// The version comes from git tags (v0.1.0 etc.), which `cargo release` creates after bumping the
// Cargo workspace version, so the JAR and the crates always carry the same number. Between tags
// the version is the next patch with -SNAPSHOT.
scmVersion {
    tag {
        prefix.set("v")
        versionSeparator.set("")
    }
    versionIncrementer("incrementPatch")
}
version = scmVersion.version

tasks.register("printVersion") {
    description = "Prints the version derived from git tags, for publish.sh."
    val v = version.toString()
    doLast { println(v) }
}

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
    testImplementation("org.junit.jupiter:junit-jupiter:6.1.3")
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

// ---------------------------------------------------------------------------
// Publishing to Maven Central (Sonatype Central Portal). Snapshots go unsigned to the snapshot
// repository; releases are signed. Sonatype credentials come from the environment (SONATYPE_USER,
// SONATYPE_PASS) or, when unset, from the Gradle properties sonatypeUsername and sonatypePassword
// in ~/.gradle/gradle.properties on a maintainer's machine. Driven by publish.sh.
// ---------------------------------------------------------------------------

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                name.set("fgkl")
                description.set("Native PairHMM, partially determined PairHMM and Smith-Waterman kernels for GATK, in Rust")
                url.set("https://github.com/fulcrumgenomics/fgkl")
                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
                developers {
                    developer {
                        id.set("tfenne")
                        name.set("Tim Fennell")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/fulcrumgenomics/fgkl.git")
                    developerConnection.set("scm:git:ssh://github.com/fulcrumgenomics/fgkl.git")
                    url.set("https://github.com/fulcrumgenomics/fgkl")
                }
            }
        }
    }
}

nexusPublishing {
    repositories {
        sonatype {
            nexusUrl.set(uri("https://ossrh-staging-api.central.sonatype.com/service/local/"))
            snapshotRepositoryUrl.set(uri("https://central.sonatype.com/repository/maven-snapshots/"))
            username.set(providers.environmentVariable("SONATYPE_USER").orElse(providers.gradleProperty("sonatypeUsername")))
            password.set(providers.environmentVariable("SONATYPE_PASS").orElse(providers.gradleProperty("sonatypePassword")))
        }
    }
}

signing {
    // Releases are signed; snapshots are not (Central rejects signed snapshots). The key comes
    // from PGP_SECRET / signingKey when given, otherwise from the local gpg with its default key
    // (or the one named by the standard `signing.gnupg.keyName` property), with gpg-agent
    // supplying the passphrase.
    val signingKey = providers.environmentVariable("PGP_SECRET").orElse(providers.gradleProperty("signingKey"))
    val signingPassword = providers.environmentVariable("PGP_PASSPHRASE").orElse(providers.gradleProperty("signingPassword"))
    if (!version.toString().endsWith("-SNAPSHOT")) {
        if (signingKey.isPresent) {
            useInMemoryPgpKeys(signingKey.get(), signingPassword.getOrElse(""))
        } else {
            useGpgCmd()
        }
        sign(publishing.publications["mavenJava"])
    }
}
