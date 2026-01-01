package dev.obrienlabs.codex.radar;

import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

@Command(name = "radar-embed", mixinStandardHelpOptions = true, version = "radar-embed 0.1.0",
    description = "Convert Doppler radar images into embeddings for generative models.")
public final class RadarEmbed implements Callable<Integer> {

    @Parameters(index = "0", description = "Path to a radar image or directory of images.")
    private Path input;

    @Option(names = "--output", description = "Output .npy file path for embeddings (metadata saved as .json).",
        defaultValue = "embeddings.npy")
    private Path output;

    @Option(names = "--batch-size", description = "Number of images per batch.", defaultValue = "16")
    private int batchSize;

    @Option(names = "--device", description = "Device to run inference on (cpu or cuda).", defaultValue = "cpu")
    private String device;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new RadarEmbed()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        if (!Files.exists(input)) {
            throw new IllegalArgumentException("Input path not found: " + input);
        }

        List<Path> imagePaths = ImagePathLoader.load(input);
        if (imagePaths.isEmpty()) {
            throw new IllegalArgumentException("No images found to embed.");
        }

        Device djlDevice = parseDevice(device);
        List<float[]> embeddings = new ArrayList<>(imagePaths.size());
        List<String> recordedPaths = new ArrayList<>(imagePaths.size());

        try (EmbeddingService service = new EmbeddingService(djlDevice)) {
            ImageFactory factory = ImageFactory.getInstance();
            for (int i = 0; i < imagePaths.size(); i += batchSize) {
                int end = Math.min(imagePaths.size(), i + batchSize);
                List<Image> batch = new ArrayList<>(end - i);
                for (int j = i; j < end; j++) {
                    Path path = imagePaths.get(j);
                    batch.add(factory.fromFile(path));
                    recordedPaths.add(path.toString());
                }
                embeddings.addAll(service.embedBatch(batch));
            }
        }

        NpyWriter.writeFloatMatrix(output, embeddings);
        MetadataWriter.writePaths(output, recordedPaths);

        System.out.printf("Saved %d embeddings to %s%n", recordedPaths.size(), output);
        return 0;
    }

    private static Device parseDevice(String value) {
        String normalized = value.toLowerCase(Locale.ROOT);
        if (normalized.startsWith("cuda") || normalized.startsWith("gpu")) {
            return Device.gpu();
        }
        return Device.cpu();
    }
}
