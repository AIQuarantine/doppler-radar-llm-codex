package com.obrienlabs.radar;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class MetadataWriter {
    private MetadataWriter() {
    }

    static void writePaths(Path output, List<String> paths) throws IOException {
        Path metadataPath = replaceExtension(output, ".json");
        Path parent = metadataPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (Writer writer = Files.newBufferedWriter(metadataPath, StandardCharsets.UTF_8)) {
            writer.write("{\"paths\":[");
            for (int i = 0; i < paths.size(); i++) {
                if (i > 0) {
                    writer.write(',');
                }
                writer.write('"');
                writer.write(escape(paths.get(i)));
                writer.write('"');
            }
            writer.write("]}");
        }
    }

    private static Path replaceExtension(Path path, String extension) {
        String fileName = path.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        String base = dot == -1 ? fileName : fileName.substring(0, dot);
        return path.resolveSibling(base + extension);
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
