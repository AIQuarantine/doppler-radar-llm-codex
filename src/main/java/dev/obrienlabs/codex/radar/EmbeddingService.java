package dev.obrienlabs.codex.radar;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.List;

final class EmbeddingService implements AutoCloseable {
    private final ZooModel<Image, float[]> model;
    private final Predictor<Image, float[]> predictor;

    EmbeddingService(Device device) throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<Image, float[]> criteria = Criteria.builder()
            .setTypes(Image.class, float[].class)
            .optApplication(Application.CV.IMAGE_EMBEDDING)
            .optFilter("backbone", "resnet50")
            .optDevice(device)
            .optTranslator(new ResNetEmbeddingTranslator())
            .build();
        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    List<float[]> embedBatch(List<Image> images) throws TranslateException {
        return predictor.batchPredict(images);
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
    }

    private static final class ResNetEmbeddingTranslator implements Translator<Image, float[]> {
        private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
        private static final float[] STD = {0.229f, 0.224f, 0.225f};
        private final Pipeline pipeline;

        ResNetEmbeddingTranslator() {
            pipeline = new Pipeline()
                .add(new Resize(256))
                .add(new CenterCrop(224, 224))
                .add(new ToTensor())
                .add(new Normalize(MEAN, STD));
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            array = pipeline.transform(new NDList(array)).singletonOrThrow();
            return new NDList(array);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            return output.toFloatArray();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
