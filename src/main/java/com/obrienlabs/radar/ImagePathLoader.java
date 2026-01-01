package com.obrienlabs.radar;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

final class ImagePathLoader {
    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(".png", ".jpg", ".jpeg", ".tif", ".tiff");

    private ImagePathLoader() {
    }

    static List<Path> load(Path root) throws IOException {
        if (Files.isRegularFile(root)) {
            return List.of(root);
        }
        List<Path> paths = new ArrayList<>();
        try (Stream<Path> stream = Files.walk(root)) {
            paths = stream
                .filter(Files::isRegularFile)
                .filter(ImagePathLoader::hasSupportedExtension)
                .sorted()
                .collect(Collectors.toList());
        }
        return paths;
    }

    private static boolean hasSupportedExtension(Path path) {
        String name = path.getFileName().toString().toLowerCase();
        for (String ext : SUPPORTED_EXTENSIONS) {
            if (name.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }
}
