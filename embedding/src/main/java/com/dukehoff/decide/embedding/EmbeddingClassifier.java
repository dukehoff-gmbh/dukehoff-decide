/*
 * Copyright 2026 Dukehoff GmbH
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dukehoff.decide.embedding;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.TaskSchema;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * A {@link ScoringBackend} trained on labelled examples: a frozen embedding model turns {@link State#render()} into
 * a vector, and one multinomial logistic regression per field on the standardised vector picks the option (a
 * "linear probe"). Unlike word counts, embeddings place texts with similar meaning close together, which helps on
 * longer texts written by people; see the README for when that is worth an embedding server.
 *
 * <pre>{@code
 * Embedder embedder = OpenAiEmbedder.builder().baseUrl("http://localhost:8081/v1").model("bge-m3").build();
 * EmbeddingClassifier model = EmbeddingClassifier.train(embedder, Triage.class, labeled);
 * model.save(Path.of("triage.model"));
 *
 * Decider decider = Decider.builder().backend(EmbeddingClassifier.load(Path.of("triage.model"), embedder)).build();
 * }</pre>
 *
 * <p>It returns log-softmax scores, so calibration, gating and the calibration report work as with any other
 * backend. Calibrate on examples the model was not trained on. The {@linkplain #modelId() model id} is derived from
 * the weights and the embedder's model id, so a retrained model gets a new calibration profile. A model only scores
 * the task it was trained for, with the same fields and options in the same order, and only with vectors from the
 * embedding model it was trained on.
 *
 * <p>Instances are immutable and thread-safe if the embedder is.
 */
public final class EmbeddingClassifier implements ScoringBackend {

    private static final int MAGIC = 0x44444345;   // "DDCE"
    private static final int FORMAT = 1;

    /** Weights per class, the last column is the bias. */
    private record Head(List<String> labels, double[][] weights) {}

    private final String modelId;
    private final Embedder embedder;
    private final String embedderId;
    private final String taskId;
    private final double[] mean, scale;
    private final Map<String, Head> heads;

    private EmbeddingClassifier(String modelId, Embedder embedder, String embedderId, String taskId, double[] mean,
                                double[] scale, Map<String, Head> heads) {
        this.embedder = embedder;
        this.embedderId = embedderId;
        this.taskId = taskId;
        this.mean = mean;
        this.scale = scale;
        this.heads = heads;
        this.modelId = modelId != null ? modelId : "embedding-" + fingerprint();
    }

    /** Trains with the default settings, see {@link Trainer}. */
    public static <T> EmbeddingClassifier train(Embedder embedder, Class<T> type, List<Calibration.Labeled<T>> examples) {
        return trainer().train(embedder, type, examples);
    }

    public static Trainer trainer() {
        return new Trainer();
    }

    @Override
    public String modelId() {
        return modelId;
    }

    /** The task this model was trained for: the output type's fully qualified name. */
    public String taskId() {
        return taskId;
    }

    /** The {@link Embedder#modelId()} of the embedding model this classifier was trained on. */
    public String embedderId() {
        return embedderId;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        return score(request.schema(), embedder.embed(request.state().render()));
    }

    /** Embeds all states in one call to the embedder, which batches them. */
    @Override
    public List<Map<String, RawScores>> scoreAll(List<ScoringRequest> requests) {
        List<float[]> vectors = embedder.embed(requests.stream().map(r -> r.state().render()).toList());
        List<Map<String, RawScores>> out = new ArrayList<>(requests.size());
        for (int i = 0; i < requests.size(); i++) out.add(score(requests.get(i).schema(), vectors.get(i)));
        return out;
    }

    private Map<String, RawScores> score(TaskSchema schema, float[] vector) {
        if (!schema.taskId().equals(taskId)) {
            throw new DecideException("Model " + modelId + " was trained for " + taskId + ", not " + schema.taskId());
        }
        if (vector.length != mean.length) {
            throw new DecideException("Model " + modelId + " expects " + mean.length + "-dimensional embeddings from "
                    + embedderId + ", got " + vector.length);
        }
        double[] x = standardise(vector, mean, scale);
        Map<String, RawScores> out = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            Head head = heads.get(f.name());
            List<String> labels = f.options().stream().map(FieldSpec.OptionSpec::label).toList();
            if (head == null || !head.labels().equals(labels)) {
                throw new DecideException("Model " + modelId + " was trained for field '" + f.name() + "' with options "
                        + (head == null ? "(none)" : head.labels()) + ", but " + taskId + " now has " + labels
                        + "; retrain it");
            }
            out.put(f.name(), new RawScores(logSoftmax(logits(head.weights(), x))));
        }
        return out;
    }

    private static double[] standardise(float[] v, double[] mean, double[] scale) {
        double[] x = new double[v.length];
        for (int i = 0; i < v.length; i++) x[i] = (v[i] - mean[i]) * scale[i];
        return x;
    }

    private static double[] logits(double[][] w, double[] x) {
        double[] z = new double[w.length];
        for (int k = 0; k < w.length; k++) {
            double s = w[k][x.length];
            for (int i = 0; i < x.length; i++) s += w[k][i] * x[i];
            z[k] = s;
        }
        return z;
    }

    private static double[] logSoftmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0), sum = 0;
        for (double v : z) sum += Math.exp(v - max);
        double log = max + Math.log(sum);
        return Arrays.stream(z).map(v -> v - log).toArray();
    }

    private static double[] softmax(double[] z) {
        double[] p = logSoftmax(z);
        for (int k = 0; k < p.length; k++) p[k] = Math.exp(p[k]);
        return p;
    }

    // ------------------------------------------------------------------------------------- training

    /** Training settings. The defaults work for embeddings with a few hundred to a few thousand dimensions. */
    public static final class Trainer {

        private int epochs = 30;
        private double learningRate = 0.05;
        private double l2 = 1e-3;
        private long seed = 42;
        private String modelId;

        private Trainer() {}

        /** Passes over the training data (default 30). */
        public Trainer epochs(int epochs) {
            if (epochs < 1) throw new IllegalArgumentException("epochs must be at least 1");
            this.epochs = epochs;
            return this;
        }

        /** AdaGrad base learning rate (default 0.05). */
        public Trainer learningRate(double learningRate) {
            if (!(learningRate > 0)) throw new IllegalArgumentException("learning rate must be positive");
            this.learningRate = learningRate;
            return this;
        }

        /** L2 regularisation per update (default 1e-3). */
        public Trainer l2(double l2) {
            if (!(l2 >= 0)) throw new IllegalArgumentException("l2 must not be negative");
            this.l2 = l2;
            return this;
        }

        /** Seed for the example order (default 42); the same vectors and seed give the same model. */
        public Trainer seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Overrides the model id derived from the weights. Calibration profiles are keyed by it. */
        public Trainer modelId(String modelId) {
            this.modelId = modelId == null || modelId.isBlank() ? null : modelId;
            return this;
        }

        public <T> EmbeddingClassifier train(Embedder embedder, Class<T> type, List<Calibration.Labeled<T>> examples) {
            return train(embedder, TaskSchema.of(type), examples);
        }

        /** For a task created with {@code decider.task(type, instructions)}, e.g. a {@code Boolean} task. */
        public <T> EmbeddingClassifier train(Embedder embedder, Class<T> type, String instructions,
                                             List<Calibration.Labeled<T>> examples) {
            return train(embedder, TaskSchema.of(type, instructions), examples);
        }

        public <T> EmbeddingClassifier train(Embedder embedder, TaskSchema schema, List<Calibration.Labeled<T>> examples) {
            return train(embedder, schema, examples.stream().map(Calibration.Labeled::state).toList(),
                    examples.stream().map(e -> schema.targetIndices(e.expected())).toList());
        }

        /**
         * For labels given as option indices rather than typed values.
         *
         * @param targets per state, the option index of every field (in {@link FieldSpec#options()} order), keyed
         *                by field name; see {@link TaskSchema#targetIndices(Object)}
         */
        public EmbeddingClassifier train(Embedder embedder, TaskSchema schema, List<State> states,
                                         List<Map<String, Integer>> targets) {
            if (states.isEmpty()) throw new IllegalArgumentException("no training examples");
            if (states.size() != targets.size()) {
                throw new IllegalArgumentException(states.size() + " states but " + targets.size() + " targets");
            }
            List<float[]> raw = embedder.embed(states.stream().map(State::render).toList());
            int dim = raw.getFirst().length;
            double[] mean = new double[dim], scale = new double[dim];
            for (float[] v : raw) {
                if (v.length != dim) throw new DecideException("Embeddings of different lengths: " + dim + " and " + v.length);
                for (int i = 0; i < dim; i++) mean[i] += v[i] / (double) raw.size();
            }
            for (float[] v : raw) for (int i = 0; i < dim; i++) scale[i] += Math.pow(v[i] - mean[i], 2) / raw.size();
            for (int i = 0; i < dim; i++) scale[i] = 1 / Math.sqrt(scale[i] + 1e-12);
            List<double[]> x = raw.stream().map(v -> standardise(v, mean, scale)).toList();

            Map<String, Head> heads = new LinkedHashMap<>();
            for (FieldSpec f : schema.fields()) {
                List<String> labels = f.options().stream().map(FieldSpec.OptionSpec::label).toList();
                int[] y = targets.stream().mapToInt(t -> {
                    Integer k = t.get(f.name());
                    if (k == null || k < 0 || k >= labels.size()) {
                        throw new IllegalArgumentException("target for '" + f.name() + "' must be 0.." + (labels.size() - 1) + ": " + k);
                    }
                    return k;
                }).toArray();
                heads.put(f.name(), new Head(labels, fit(x, y, labels.size())));
            }
            return new EmbeddingClassifier(modelId, embedder, embedder.modelId(), schema.taskId(), mean, scale, heads);
        }

        private double[][] fit(List<double[]> x, int[] y, int classes) {
            int dim = x.getFirst().length;
            double[][] w = new double[classes][dim + 1], g2 = new double[classes][dim + 1];
            int[] order = new int[x.size()];
            Arrays.setAll(order, i -> i);
            Random random = new Random(seed);
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = order.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1), t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
                for (int n : order) {
                    double[] xi = x.get(n);
                    double[] p = softmax(logits(w, xi));
                    for (int k = 0; k < classes; k++) {
                        double err = p[k] - (k == y[n] ? 1 : 0);
                        double[] wk = w[k], gk = g2[k];
                        for (int i = 0; i < dim; i++) {
                            double g = err * xi[i] + l2 * wk[i];
                            gk[i] += g * g;
                            wk[i] -= learningRate * g / (Math.sqrt(gk[i]) + 1e-8);
                        }
                        gk[dim] += err * err;
                        wk[dim] -= learningRate * err / (Math.sqrt(gk[dim]) + 1e-8);
                    }
                }
            }
            return w;
        }
    }

    // ------------------------------------------------------------------------------------- persistence

    /** Writes the model (gzip-compressed). The embedding model itself is not part of it. */
    public void save(Path file) {
        try (OutputStream out = Files.newOutputStream(file)) {
            save(out);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write " + file, e);
        }
    }

    /** Writes the model to {@code out} and leaves it open. */
    public void save(OutputStream out) throws IOException {
        GZIPOutputStream gzip = new GZIPOutputStream(out);
        DataOutputStream data = new DataOutputStream(new BufferedOutputStream(gzip));
        data.writeInt(MAGIC);
        data.writeInt(FORMAT);
        data.writeUTF(modelId);
        writeBody(data);
        data.flush();
        gzip.finish();
    }

    /**
     * @param embedder must have the {@link Embedder#modelId()} the model was trained with
     */
    public static EmbeddingClassifier load(Path file, Embedder embedder) {
        try (InputStream in = Files.newInputStream(file)) {
            return load(in, embedder);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read " + file, e);
        }
    }

    /** Reads a model written by {@link #save(OutputStream)}, e.g. from a classpath resource. */
    public static EmbeddingClassifier load(InputStream in, Embedder embedder) throws IOException {
        DataInputStream data = new DataInputStream(new BufferedInputStream(new GZIPInputStream(in)));
        if (data.readInt() != MAGIC) throw new IOException("Not a dukehoff-decide embedding classifier");
        int format = data.readInt();
        if (format != FORMAT) throw new IOException("Unsupported model format " + format + ", expected " + FORMAT);
        String modelId = data.readUTF();
        String embedderId = data.readUTF();
        if (!embedderId.equals(embedder.modelId())) {
            throw new DecideException("Model " + modelId + " was trained on embeddings from " + embedderId
                    + ", not " + embedder.modelId());
        }
        String taskId = data.readUTF();
        int dim = data.readInt();
        double[] mean = new double[dim], scale = new double[dim];
        for (int i = 0; i < dim; i++) mean[i] = data.readDouble();
        for (int i = 0; i < dim; i++) scale[i] = data.readDouble();
        Map<String, Head> heads = new LinkedHashMap<>();
        for (int fields = data.readInt(); fields > 0; fields--) {
            String name = data.readUTF();
            List<String> labels = new ArrayList<>();
            for (int n = data.readInt(); n > 0; n--) labels.add(data.readUTF());
            double[][] w = new double[labels.size()][dim + 1];
            for (double[] row : w) for (int i = 0; i <= dim; i++) row[i] = data.readDouble();
            heads.put(name, new Head(List.copyOf(labels), w));
        }
        return new EmbeddingClassifier(modelId, embedder, embedderId, taskId, mean, scale, heads);
    }

    /** Everything but the model id; also the input of {@link #fingerprint()}. */
    private void writeBody(DataOutputStream data) throws IOException {
        data.writeUTF(embedderId);
        data.writeUTF(taskId);
        data.writeInt(mean.length);
        for (double v : mean) data.writeDouble(v);
        for (double v : scale) data.writeDouble(v);
        data.writeInt(heads.size());
        for (Map.Entry<String, Head> e : heads.entrySet()) {
            data.writeUTF(e.getKey());
            data.writeInt(e.getValue().labels().size());
            for (String l : e.getValue().labels()) data.writeUTF(l);
            for (double[] row : e.getValue().weights()) for (double v : row) data.writeDouble(v);
        }
    }

    private String fingerprint() {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            try (DataOutputStream data = new DataOutputStream(new BufferedOutputStream(
                    new DigestOutputStream(OutputStream.nullOutputStream(), sha)))) {
                writeBody(data);
            }
            return HexFormat.of().formatHex(sha.digest(), 0, 8);
        } catch (NoSuchAlgorithmException | IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public String toString() {
        return "EmbeddingClassifier[" + modelId + ", " + embedderId + ", " + taskId + ", fields " + heads.keySet() + "]";
    }
}
