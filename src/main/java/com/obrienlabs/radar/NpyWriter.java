package com.obrienlabs.radar;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class NpyWriter {
    private static final byte[] MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

    private NpyWriter() {
    }

    static void writeFloatMatrix(Path output, List<float[]> rows) throws IOException {
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("No embeddings to write.");
        }

        int rowCount = rows.size();
        int colCount = rows.get(0).length;
        for (float[] row : rows) {
            if (row.length != colCount) {
                throw new IllegalArgumentException("Inconsistent embedding dimensions.");
            }
        }

        String header = String.format("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
            rowCount, colCount);
        byte[] headerBytes = buildHeader(header);

        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (OutputStream stream = Files.newOutputStream(output)) {
            stream.write(MAGIC);
            stream.write(new byte[]{1, 0});
            stream.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) headerBytes.length).array());
            stream.write(headerBytes);
            ByteBuffer buffer = ByteBuffer.allocate(rowCount * colCount * Float.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
            for (float[] row : rows) {
                for (float value : row) {
                    buffer.putFloat(value);
                }
            }
            stream.write(buffer.array());
        }
    }

    private static byte[] buildHeader(String header) {
        int preamble = MAGIC.length + 2 + 2;
        int headerLen = header.length() + 1;
        int padding = (16 - ((preamble + headerLen) % 16)) % 16;
        String padded = header + " ".repeat(padding) + "\n";
        return padded.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    }
}
