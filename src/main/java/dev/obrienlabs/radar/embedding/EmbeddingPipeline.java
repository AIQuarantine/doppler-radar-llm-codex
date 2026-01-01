package dev.obrienlabs.radar.embedding;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Pipeline;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class EmbeddingPipeline {
    private static final Set<String> SUPPORTED_EXTENSIONS =
            Set.of(".png", ".jpg", ".jpeg", ".tif", ".tiff");

    public List<Path> loadImagePaths(Path input) throws IOException {
        if (Files.notExists(input)) {
            throw new IllegalArgumentException("Input path not found: " + input);
        }
        if (Files.isRegularFile(input)) {
            return List.of(input);
        }
        try (Stream<Path> stream = Files.walk(input)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path -> SUPPORTED_EXTENSIONS.contains(extensionOf(path)))
                    .sorted(Comparator.naturalOrder())
                    .collect(Collectors.toList());
        }
    }

    public EmbeddingResult embed(Path input, String device) throws IOException, ModelException, TranslateException {
        List<Path> imagePaths = loadImagePaths(input);
        if (imagePaths.isEmpty()) {
            throw new IllegalArgumentException("No images found to embed.");
        }

        Criteria<Image, float[]> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, float[].class)
                .optFilter("layers", "50")
                .optTranslator(new LogitsTranslator())
                .optDevice(ai.djl.Device.fromName(device))
                .build();

        List<float[]> embeddings = new ArrayList<>();
        List<String> paths = new ArrayList<>();

        try (ZooModel<Image, float[]> model = ModelZoo.loadModel(criteria);
             Predictor<Image, float[]> predictor = model.newPredictor()) {
            ImageFactory factory = ImageFactory.getInstance();
            for (Path path : imagePaths) {
                Image image = factory.fromFile(path);
                float[] vector = predictor.predict(image);
                embeddings.add(vector);
                paths.add(path.toString());
            }
        }

        float[][] array = embeddings.toArray(new float[0][]);
        return new EmbeddingResult(array, paths);
    }

    public void save(EmbeddingResult result, Path output) throws IOException {
        if (output.getParent() != null) {
            Files.createDirectories(output.getParent());
        }
        EmbeddingWriter.writeBinary(output, result.embeddings());
        int columns = result.embeddings().length == 0 ? 0 : result.embeddings()[0].length;
        EmbeddingWriter.writeMetadata(output, columns, result.paths());
    }

    private static String extensionOf(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        int index = name.lastIndexOf('.');
        if (index < 0) {
            return "";
        }
        return name.substring(index);
    }

    private static final class LogitsTranslator implements Translator<Image, float[]> {
        private final Pipeline pipeline;

        private LogitsTranslator() {
            pipeline = new Pipeline()
                    .add(new Resize(256))
                    .add(new CenterCrop(224, 224))
                    .add(new ToTensor())
                    .add(new Normalize(
                            new float[]{0.485f, 0.456f, 0.406f},
                            new float[]{0.229f, 0.224f, 0.225f}
                    ));
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager());
            array = pipeline.transform(array);
            array = array.expandDims(0);
            return new NDList(array);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            NDArray squeezed = output.squeeze(0);
            return squeezed.toFloatArray();
        }

        @Override
        public void close() {
            // No resources to clean up.
        }
    }
}
