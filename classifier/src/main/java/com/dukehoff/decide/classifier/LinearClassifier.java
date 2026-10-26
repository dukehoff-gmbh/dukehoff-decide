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
package com.dukehoff.decide.classifier;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.classifier.Features.Sparse;
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
 * A {@link ScoringBackend} trained on labelled examples instead of prompting a language model: one multinomial
 * logistic regression per field on hashed TF-IDF features of {@link com.dukehoff.decide.State#render()} (word
 * 1-2-grams and character 3-5-grams). No model server and no dependencies; scoring takes well under a millisecond.
 *
 * <pre>{@code
 * LinearClassifier model = LinearClassifier.train(Triage.class, labeled);   // List<Calibration.Labeled<Triage>>
 * model.save(Path.of("triage.model"));
 *
 * Decider decider = Decider.builder().backend(LinearClassifier.load(Path.of("triage.model"))).build();
 * }</pre>
 *
 * <p>It returns log-softmax scores, so calibration, gating and the calibration report work as with any other
 * backend. Calibrate on examples the model was not trained on: its scores on its own training data are
 * overconfident. Needs a few thousand labelled examples to beat a prompted language model; see the README.
 *
 * <p>The {@linkplain #modelId() model id} is derived from the trained weights, so a retrained model gets a new id
 * and its old calibration profile is not reused. A model only scores the task it was trained for, with the same
 * fields and options in the same order; otherwise it throws a {@link DecideException} that asks for retraining.
 *
 * <p>Instances are immutable and thread-safe.
 */
public final class LinearClassifier implements ScoringBackend {

    private static final int MAGIC = 0x44444C43;   // "DDLC"
    private static final int FORMAT = 1;

    /** Weights per class, the last column is the bias. */
    private record Head(List<String> labels, float[][] weights) {}

    private final String modelId;
    private final String taskId;
    private final int hashBits;
    private final float[] idf;
    private final Map<String, Head> heads;

    private LinearClassifier(String modelId, String taskId, int hashBits, float[] idf, Map<String, Head> heads) {
        this.taskId = taskId;
        this.hashBits = hashBits;
        this.idf = idf;
        this.heads = heads;
        this.modelId = modelId != null ? modelId : "linear-" + fingerprint();
    }

    /** Trains with the default settings, see {@link Trainer}. */
    public static <T> LinearClassifier train(Class<T> type, List<Calibration.Labeled<T>> examples) {
        return trainer().train(type, examples);
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

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        TaskSchema schema = request.schema();
        if (!schema.taskId().equals(taskId)) {
            throw new DecideException("Model " + modelId + " was trained for " + taskId + ", not " + schema.taskId());
        }
        Sparse x = Features.tfidf(Features.counts(request.state().render(), idf.length), idf);
        Map<String, RawScores> out = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            Head head = heads.get(f.name());
            List<String> labels = f.options().stream().map(FieldSpec.OptionSpec::label).toList();
            if (head == null || !head.labels().equals(labels)) {
                throw new DecideException("Model " + modelId + " was trained for field '" + f.name() + "' with options "
                        + (head == null ? "(none)" : head.labels()) + ", but " + taskId + " now has " + labels
                        + "; retrain it");
            }
            out.put(f.name(), new RawScores(Features.logSoftmax(logits(head.weights(), x))));
        }
        return out;
    }

    private static double[] logits(float[][] w, Sparse x) {
        double[] z = new double[w.length];
        int bias = w[0].length - 1;
        for (int k = 0; k < w.length; k++) {
            double s = w[k][bias];
            for (int j = 0; j < x.index().length; j++) s += w[k][x.index()[j]] * x.value()[j];
            z[k] = s;
        }
        return z;
    }

    // ------------------------------------------------------------------------------------- training

    /** Training settings. The defaults work for texts from a few words to a few pages. */
    public static final class Trainer {

        private int epochs = 15;
        private double learningRate = 0.5;
        private double l2 = 1e-6;
        private int hashBits = 18;
        private long seed = 42;
        private String modelId;

        private Trainer() {}

        /** Passes over the training data (default 15). */
        public Trainer epochs(int epochs) {
            if (epochs < 1) throw new IllegalArgumentException("epochs must be at least 1");
            this.epochs = epochs;
            return this;
        }

        /** AdaGrad base learning rate (default 0.5). */
        public Trainer learningRate(double learningRate) {
            if (!(learningRate > 0)) throw new IllegalArgumentException("learning rate must be positive");
            this.learningRate = learningRate;
            return this;
        }

        /** L2 regularisation per update (default 1e-6). */
        public Trainer l2(double l2) {
            if (!(l2 >= 0)) throw new IllegalArgumentException("l2 must not be negative");
            this.l2 = l2;
            return this;
        }

        /** Features are hashed into 2^bits buckets (default 18). Memory per field: options × 2^bits floats. */
        public Trainer hashBits(int bits) {
            if (bits < 4 || bits > 24) throw new IllegalArgumentException("hash bits must be between 4 and 24");
            this.hashBits = bits;
            return this;
        }

        /** Seed for the example order (default 42); the same data and seed give the same model. */
        public Trainer seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Overrides the model id derived from the weights. Calibration profiles are keyed by it. */
        public Trainer modelId(String modelId) {
            this.modelId = modelId == null || modelId.isBlank() ? null : modelId;
            return this;
        }

        public <T> LinearClassifier train(Class<T> type, List<Calibration.Labeled<T>> examples) {
            return train(TaskSchema.of(type), examples);
        }

        /** For a task created with {@code decider.task(type, instructions)}, e.g. a {@code Boolean} task. */
        public <T> LinearClassifier train(Class<T> type, String instructions, List<Calibration.Labeled<T>> examples) {
            return train(TaskSchema.of(type, instructions), examples);
        }

        public <T> LinearClassifier train(TaskSchema schema, List<Calibration.Labeled<T>> examples) {
            return train(schema, examples.stream().map(Calibration.Labeled::state).toList(),
                    examples.stream().map(e -> schema.targetIndices(e.expected())).toList());
        }

        /**
         * For labels given as option indices rather than typed values.
         *
         * @param targets per state, the option index of every field (in {@link FieldSpec#options()} order), keyed
         *                by field name; see {@link TaskSchema#targetIndices(Object)}
         */
        public LinearClassifier train(TaskSchema schema, List<State> states, List<Map<String, Integer>> targets) {
            if (states.isEmpty()) throw new IllegalArgumentException("no training examples");
            if (states.size() != targets.size()) {
                throw new IllegalArgumentException(states.size() + " states but " + targets.size() + " targets");
            }
            int dim = 1 << hashBits;
            List<Map<Integer, Double>> counts = states.stream().map(s -> Features.counts(s.render(), dim)).toList();
            double[] df = new double[dim];
            for (Map<Integer, Double> c : counts) c.keySet().forEach(i -> df[i]++);
            float[] idf = new float[dim];
            for (int i = 0; i < dim; i++) idf[i] = (float) (Math.log((1 + states.size()) / (1 + df[i])) + 1);
            List<Sparse> x = counts.stream().map(c -> Features.tfidf(c, idf)).toList();

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
                heads.put(f.name(), new Head(labels, fit(x, y, labels.size(), dim)));
            }
            return new LinearClassifier(modelId, schema.taskId(), hashBits, idf, heads);
        }

        private float[][] fit(List<Sparse> x, int[] y, int classes, int dim) {
            double[][] w = new double[classes][dim + 1];
            double[][] g2 = new double[classes][dim + 1];
            int[] order = new int[x.size()];
            Arrays.setAll(order, i -> i);
            Random random = new Random(seed);
            for (int epoch = 0; epoch < epochs; epoch++) {
                shuffle(order, random);
                for (int n : order) {
                    Sparse xi = x.get(n);
                    double[] p = Features.softmax(logits(w, xi));
                    for (int k = 0; k < classes; k++) {
                        double err = p[k] - (k == y[n] ? 1 : 0);
                        for (int j = 0; j < xi.index().length; j++) {
                            int i = xi.index()[j];
                            update(w[k], g2[k], i, err * xi.value()[j] + l2 * w[k][i]);
                        }
                        update(w[k], g2[k], dim, err);
                    }
                }
            }
            float[][] out = new float[classes][dim + 1];
            for (int k = 0; k < classes; k++) {
                for (int i = 0; i <= dim; i++) out[k][i] = (float) w[k][i];
            }
            return out;
        }

        private void update(double[] w, double[] g2, int i, double g) {
            g2[i] += g * g;
            w[i] -= learningRate * g / (Math.sqrt(g2[i]) + 1e-8);
        }

        private static double[] logits(double[][] w, Sparse x) {
            double[] z = new double[w.length];
            int bias = w[0].length - 1;
            for (int k = 0; k < w.length; k++) {
                double s = w[k][bias];
                for (int j = 0; j < x.index().length; j++) s += w[k][x.index()[j]] * x.value()[j];
                z[k] = s;
            }
            return z;
        }

        private static void shuffle(int[] a, Random random) {
            for (int i = a.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
    }

    // ------------------------------------------------------------------------------------- persistence

    /** Writes the model (gzip-compressed, only non-zero weights). */
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

    public static LinearClassifier load(Path file) {
        try (InputStream in = Files.newInputStream(file)) {
            return load(in);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read " + file, e);
        }
    }

    /** Reads a model written by {@link #save(OutputStream)}, e.g. from a classpath resource. */
    public static LinearClassifier load(InputStream in) throws IOException {
        DataInputStream data = new DataInputStream(new BufferedInputStream(new GZIPInputStream(in)));
        if (data.readInt() != MAGIC) throw new IOException("Not a dukehoff-decide linear classifier");
        int format = data.readInt();
        if (format != FORMAT) throw new IOException("Unsupported model format " + format + ", expected " + FORMAT);
        String modelId = data.readUTF();
        String taskId = data.readUTF();
        int hashBits = data.readInt();
        int dim = 1 << hashBits;
        float[] idf = new float[dim];
        Arrays.fill(idf, data.readFloat());
        for (int n = data.readInt(); n > 0; n--) {
            int i = data.readInt();
            idf[i] = data.readFloat();
        }
        Map<String, Head> heads = new LinkedHashMap<>();
        for (int fields = data.readInt(); fields > 0; fields--) {
            String name = data.readUTF();
            List<String> labels = new ArrayList<>();
            for (int n = data.readInt(); n > 0; n--) labels.add(data.readUTF());
            float[][] w = new float[labels.size()][dim + 1];
            for (int n = data.readInt(); n > 0; n--) {
                int i = data.readInt();
                for (float[] row : w) row[i] = data.readFloat();
            }
            for (float[] row : w) row[dim] = data.readFloat();
            heads.put(name, new Head(List.copyOf(labels), w));
        }
        return new LinearClassifier(modelId, taskId, hashBits, idf, heads);
    }

    /** Everything but the model id; also the input of {@link #fingerprint()}. */
    private void writeBody(DataOutputStream data) throws IOException {
        int dim = idf.length;
        data.writeUTF(taskId);
        data.writeInt(hashBits);
        float unseen = unseenIdf(idf);
        data.writeFloat(unseen);
        int seen = 0;
        for (float v : idf) if (v != unseen) seen++;
        data.writeInt(seen);
        for (int i = 0; i < dim; i++) {
            if (idf[i] != unseen) {
                data.writeInt(i);
                data.writeFloat(idf[i]);
            }
        }
        data.writeInt(heads.size());
        for (Map.Entry<String, Head> e : heads.entrySet()) {
            data.writeUTF(e.getKey());
            List<String> labels = e.getValue().labels();
            float[][] w = e.getValue().weights();
            data.writeInt(labels.size());
            for (String l : labels) data.writeUTF(l);
            int nonZero = 0;
            for (int i = 0; i < dim; i++) if (nonZero(w, i)) nonZero++;
            data.writeInt(nonZero);
            for (int i = 0; i < dim; i++) {
                if (!nonZero(w, i)) continue;
                data.writeInt(i);
                for (float[] row : w) data.writeFloat(row[i]);
            }
            for (float[] row : w) data.writeFloat(row[dim]);
        }
    }

    private static boolean nonZero(float[][] w, int i) {
        for (float[] row : w) if (row[i] != 0) return true;
        return false;
    }

    /** The idf of features that never occurred in training, which is most of the buckets. */
    private static float unseenIdf(float[] idf) {
        float max = idf[0];
        for (float v : idf) max = Math.max(max, v);
        return max;
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
        return "LinearClassifier[" + modelId + ", " + taskId + ", fields " + heads.keySet() + "]";
    }
}
