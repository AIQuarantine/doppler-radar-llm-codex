package dev.obrienlabs.radar.embedding;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class EmbeddingWriter {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private EmbeddingWriter() {
    }

    public static void writeBinary(Path output, float[][] embeddings) throws IOException {
        try (BufferedOutputStream stream = new BufferedOutputStream(Files.newOutputStream(output))) {
            int columns = embeddings.length == 0 ? 0 : embeddings[0].length;
            ByteBuffer buffer = ByteBuffer.allocate(4 * columns);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            for (float[] vector : embeddings) {
                buffer.clear();
                for (float value : vector) {
                    buffer.putFloat(value);
                }
                stream.write(buffer.array());
            }
        }
    }

    public static void writeMetadata(Path output, int columns, List<String> paths) throws IOException {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("embedding_format", "float32_le");
        root.put("output", output.getFileName().toString());
        root.put("rows", paths.size());
        root.put("columns", columns);
        ArrayNode array = root.putArray("paths");
        for (String path : paths) {
            array.add(path);
        }
        Path metadata = metadataPath(output);
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(metadata.toFile(), root);
    }

    private static Path metadataPath(Path output) {
        String name = output.getFileName().toString();
        return output.resolveSibling(name + ".json");
    }
}
