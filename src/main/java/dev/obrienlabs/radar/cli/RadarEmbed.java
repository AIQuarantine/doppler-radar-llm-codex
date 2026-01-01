package dev.obrienlabs.radar.cli;

import ai.djl.ModelException;
import ai.djl.translate.TranslateException;
import dev.obrienlabs.radar.embedding.EmbeddingPipeline;
import dev.obrienlabs.radar.embedding.EmbeddingResult;
import java.io.IOException;
import java.nio.file.Path;

public final class RadarEmbed {
    public static void main(String[] args) {
        try {
            Options options = Options.parse(args);
            EmbeddingPipeline pipeline = new EmbeddingPipeline();
            EmbeddingResult result = pipeline.embed(options.input(), options.device());
            pipeline.save(result, options.output());
            System.out.printf("Saved %d embeddings to %s%n", result.size(), options.output());
        } catch (IllegalArgumentException | IOException | ModelException | TranslateException error) {
            System.err.println("Error: " + error.getMessage());
            System.exit(1);
        }
    }

    private record Options(Path input, Path output, String device) {
        static Options parse(String[] args) {
            if (args.length == 0) {
                printUsageAndExit();
            }
            Path input = Path.of(args[0]);
            Path output = Path.of("embeddings.bin");
            String device = "cpu";
            for (int i = 1; i < args.length; i++) {
                String arg = args[i];
                if ("--output".equals(arg) && i + 1 < args.length) {
                    output = Path.of(args[++i]);
                } else if ("--device".equals(arg) && i + 1 < args.length) {
                    device = args[++i];
                } else if ("--help".equals(arg) || "-h".equals(arg)) {
                    printUsageAndExit();
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + arg);
                }
            }
            return new Options(input, output, device);
        }

        private static void printUsageAndExit() {
            System.out.println("Usage: radar-embed <input> [--output embeddings.bin] [--device cpu]");
            System.out.println("  input: path to image or directory of images");
            System.exit(0);
        }
    }
}
