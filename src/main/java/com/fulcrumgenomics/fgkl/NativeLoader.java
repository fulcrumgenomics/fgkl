package com.fulcrumgenomics.fgkl;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.EnumSet;

/**
 * Extracts the platform-specific fgkl native library from the JAR and loads it once per JVM.
 *
 * <p>The library is looked up at {@code /native/<os>-<arch>/<libfgkl>} inside the JAR. Setting the
 * system property {@code fgkl.library.path} to a directory loads the library from there instead,
 * which is how a locally built library is tested without repackaging.
 *
 * <p>A JVM can load a given native library into one class loader only; a second class loader
 * (for example a Spark executor's) asking for it gets the JVM's own {@code UnsatisfiedLinkError}.
 */
public final class NativeLoader {
    private static final String LIBRARY = "fgkl";
    private static final String PATH_PROPERTY = "fgkl.library.path";
    private static volatile boolean loaded = false;

    private NativeLoader() {}

    /** True once the library has been loaded in this JVM. */
    public static boolean isLoaded() {
        return loaded;
    }

    /**
     * Loads the native library, extracting it under {@code tempDir} (or the default temporary
     * directory when null) if it has to come out of the JAR.
     *
     * @throws UnsatisfiedLinkError if no library exists for this platform or it fails to load
     */
    public static synchronized void load(File tempDir) {
        if (loaded) return;
        String override = System.getProperty(PATH_PROPERTY);
        if (override != null) {
            System.load(Paths.get(override, System.mapLibraryName(LIBRARY)).toAbsolutePath().toString());
            loaded = true;
            return;
        }
        String platform = detectPlatform();
        String extension = platform.startsWith("osx") ? "dylib" : platform.startsWith("linux") ? "so" : "dll";
        String fileName = platform.startsWith("windows") ? LIBRARY + "." + extension : "lib" + LIBRARY + "." + extension;
        String resource = "/native/" + platform + "/" + fileName;
        try (InputStream in = NativeLoader.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new UnsatisfiedLinkError("No fgkl native library for platform " + platform + " (looked for " + resource + " in the JAR)");
            }
            Path dir = tempDir == null ? null : tempDir.toPath();
            Path temp;
            try {
                temp = createTempFile(dir, extension, PosixFilePermissions.asFileAttribute(
                        EnumSet.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE)));
            } catch (UnsupportedOperationException e) {
                temp = createTempFile(dir, extension);
            }
            Files.copy(in, temp, StandardCopyOption.REPLACE_EXISTING);
            try {
                System.load(temp.toAbsolutePath().toString());
            } catch (UnsatisfiedLinkError e) {
                UnsatisfiedLinkError wrapped = new UnsatisfiedLinkError("Failed to load the fgkl native library from " + temp
                        + " (a noexec temp directory is the usual cause: set -Djava.io.tmpdir, pass a tempDir, or point -D"
                        + PATH_PROPERTY + " at a directory holding the library): " + e.getMessage());
                wrapped.initCause(e);
                throw wrapped;
            }
            try {
                Files.delete(temp);
            } catch (IOException ignored) {
                temp.toFile().deleteOnExit();
            }
        } catch (IOException e) {
            UnsatisfiedLinkError wrapped = new UnsatisfiedLinkError("Failed to extract the fgkl native library: " + e.getMessage());
            wrapped.initCause(e);
            throw wrapped;
        }
        loaded = true;
    }

    private static Path createTempFile(Path dir, String extension, java.nio.file.attribute.FileAttribute<?>... attrs) throws IOException {
        return dir == null ? Files.createTempFile("fgkl-", "." + extension, attrs) : Files.createTempFile(dir, "fgkl-", "." + extension, attrs);
    }

    /** The platform string used in resource paths, e.g. {@code linux-x86_64} or {@code osx-aarch64}. */
    public static String detectPlatform() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        String os;
        if (osName.contains("mac") || osName.contains("darwin")) {
            os = "osx";
        } else if (osName.contains("linux")) {
            os = "linux";
        } else if (osName.contains("win")) {
            os = "windows";
        } else {
            throw new UnsatisfiedLinkError("Unsupported operating system: " + osName);
        }
        String archName = System.getProperty("os.arch", "").toLowerCase();
        String arch;
        if (archName.equals("amd64") || archName.equals("x86_64")) {
            arch = "x86_64";
        } else if (archName.equals("aarch64") || archName.equals("arm64")) {
            arch = "aarch64";
        } else {
            throw new UnsatisfiedLinkError("Unsupported architecture: " + archName);
        }
        return os + "-" + arch;
    }
}
