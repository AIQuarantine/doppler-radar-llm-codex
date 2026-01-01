package dev.obrienlabs.radar.embedding;

import java.util.List;

public record EmbeddingResult(float[][] embeddings, List<String> paths) {
    public int size() {
        return embeddings == null ? 0 : embeddings.length;
    }
}
