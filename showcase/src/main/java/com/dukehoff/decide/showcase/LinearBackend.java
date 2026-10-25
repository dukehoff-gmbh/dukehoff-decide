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
package com.dukehoff.decide.showcase;

import com.dukehoff.decide.State;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.TaskSchema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Experiment: no language model, one multinomial logistic regression per field on TF-IDF features
 * (word 1-2-grams and character 3-5-grams, hashed into {@value #DIM} buckets, L2-normalised).
 * Trained with AdaGrad SGD and L2 regularisation. Returns log-softmax scores, so calibration,
 * gating and the report work unchanged.
 */
final class LinearBackend implements ScoringBackend {

    static final int DIM = 1 << 18;

    record Sparse(int[] index, double[] value) {}

    private final String modelId;
    private final double[] idf;
    private final Map<String, double[][]> weights;

    private LinearBackend(String modelId, double[] idf, Map<String, double[][]> weights) {
        this.modelId = modelId;
        this.idf = idf;
        this.weights = weights;
    }

    /**
     * @param targets per example, the option index of each field (in {@link FieldSpec#options()} order)
     */
    static LinearBackend train(String modelId, TaskSchema schema, List<State> states,
                               List<Map<String, Integer>> targets, long seed) {
        List<Map<Integer, Double>> counts = states.stream().map(s -> counts(s.render())).toList();
        double[] df = new double[DIM];
        for (Map<Integer, Double> c : counts) c.keySet().forEach(i -> df[i]++);
        double[] idf = new double[DIM];
        for (int i = 0; i < DIM; i++) idf[i] = Math.log((1 + states.size()) / (1 + df[i])) + 1;
        List<Sparse> x = counts.stream().map(c -> tfidf(c, idf)).toList();

        Map<String, double[][]> weights = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            int[] y = targets.stream().mapToInt(t -> t.get(f.name())).toArray();
            weights.put(f.name(), fit(x, y, f.options().size(), seed));
        }
        return new LinearBackend(modelId, idf, weights);
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        Sparse x = tfidf(counts(request.state().render()), idf);
        Map<String, RawScores> out = new LinkedHashMap<>();
        for (FieldSpec f : request.schema().fields()) {
            out.put(f.name(), new RawScores(logSoftmax(logits(weights.get(f.name()), x))));
        }
        return out;
    }

    // ------------------------------------------------------------------------------------- training

    static final int EPOCHS = 15;
    static final double RATE = 0.5, L2 = 1e-6;

    /** @return weights per class, the last column of each row is the bias */
    static double[][] fit(List<Sparse> x, int[] y, int classes, long seed) {
        double[][] w = new double[classes][DIM + 1];
        double[][] g2 = new double[classes][DIM + 1];
        int[] order = new int[x.size()];
        Arrays.setAll(order, i -> i);
        Random random = new Random(seed);
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            shuffle(order, random);
            for (int n : order) {
                Sparse xi = x.get(n);
                double[] p = softmax(logits(w, xi));
                for (int k = 0; k < classes; k++) {
                    double err = p[k] - (k == y[n] ? 1 : 0);
                    for (int j = 0; j < xi.index().length; j++) {
                        int i = xi.index()[j];
                        update(w[k], g2[k], i, err * xi.value()[j] + L2 * w[k][i]);
                    }
                    update(w[k], g2[k], DIM, err);
                }
            }
        }
        return w;
    }

    private static void update(double[] w, double[] g2, int i, double g) {
        g2[i] += g * g;
        w[i] -= RATE * g / (Math.sqrt(g2[i]) + 1e-8);
    }

    private static void shuffle(int[] a, Random random) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1), t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    static double[] logits(double[][] w, Sparse x) {
        double[] z = new double[w.length];
        for (int k = 0; k < w.length; k++) {
            double s = w[k][DIM];
            for (int j = 0; j < x.index().length; j++) s += w[k][x.index()[j]] * x.value()[j];
            z[k] = s;
        }
        return z;
    }

    static double[] softmax(double[] z) {
        double[] p = logSoftmax(z);
        for (int k = 0; k < p.length; k++) p[k] = Math.exp(p[k]);
        return p;
    }

    static double[] logSoftmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0), sum = 0;
        for (double v : z) sum += Math.exp(v - max);
        double log = max + Math.log(sum);
        return Arrays.stream(z).map(v -> v - log).toArray();
    }

    // ------------------------------------------------------------------------------------- features

    /** Hashed n-gram counts. */
    static Map<Integer, Double> counts(String text) {
        String t = text.toLowerCase(Locale.ROOT);
        Map<Integer, Double> c = new TreeMap<>();
        List<String> words = new ArrayList<>(Arrays.asList(t.split("[^\\p{L}\\p{N}{}]+")));
        words.removeIf(String::isEmpty);
        for (int i = 0; i < words.size(); i++) {
            add(c, "w:" + words.get(i));
            if (i > 0) add(c, "b:" + words.get(i - 1) + ' ' + words.get(i));
        }
        String padded = ' ' + String.join(" ", words) + ' ';
        for (int n = 3; n <= 5; n++) {
            for (int i = 0; i + n <= padded.length(); i++) add(c, "c:" + padded.substring(i, i + n));
        }
        return c;
    }

    private static void add(Map<Integer, Double> c, String feature) {
        c.merge(Math.floorMod(feature.hashCode() * 0x9E3779B1, DIM), 1.0, Double::sum);
    }

    /** Sublinear TF times IDF, L2-normalised. */
    static Sparse tfidf(Map<Integer, Double> counts, double[] idf) {
        int[] index = new int[counts.size()];
        double[] value = new double[counts.size()];
        int j = 0;
        double norm = 0;
        for (Map.Entry<Integer, Double> e : counts.entrySet()) {
            index[j] = e.getKey();
            value[j] = (1 + Math.log(e.getValue())) * idf[e.getKey()];
            norm += value[j] * value[j];
            j++;
        }
        norm = Math.sqrt(norm);
        for (j = 0; j < value.length; j++) value[j] /= norm == 0 ? 1 : norm;
        return new Sparse(index, value);
    }
}
