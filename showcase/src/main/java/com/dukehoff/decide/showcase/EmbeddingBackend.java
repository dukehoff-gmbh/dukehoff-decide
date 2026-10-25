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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Experiment: a frozen sentence-embedding model (any OpenAI-compatible {@code /embeddings} endpoint) plus one
 * multinomial logistic regression per field on the standardised vectors (a "linear probe"). Trained with AdaGrad
 * SGD and L2 regularisation. Returns log-softmax scores, so calibration, gating and the report work unchanged.
 */
final class EmbeddingBackend implements ScoringBackend {

    static final int EPOCHS = 30;
    static final double RATE = 0.05, L2 = 1e-3;

    private final String modelId;
    private final Embedder embedder;
    private final double[] mean, scale;
    private final Map<String, double[][]> weights;

    private EmbeddingBackend(String modelId, Embedder embedder, double[] mean, double[] scale,
                             Map<String, double[][]> weights) {
        this.modelId = modelId;
        this.embedder = embedder;
        this.mean = mean;
        this.scale = scale;
        this.weights = weights;
    }

    /**
     * @param targets per example, the option index of each field (in {@link FieldSpec#options()} order)
     */
    static EmbeddingBackend train(String modelId, Embedder embedder, TaskSchema schema, List<State> states,
                                  List<Map<String, Integer>> targets, long seed) {
        embedder.embedAll(states.stream().map(State::render).toList());
        List<float[]> raw = states.stream().map(s -> embedder.embed(s.render())).toList();
        int dim = raw.getFirst().length;
        double[] mean = new double[dim], scale = new double[dim];
        for (float[] v : raw) for (int i = 0; i < dim; i++) mean[i] += v[i] / (double) raw.size();
        for (float[] v : raw) for (int i = 0; i < dim; i++) scale[i] += Math.pow(v[i] - mean[i], 2) / raw.size();
        for (int i = 0; i < dim; i++) scale[i] = 1 / Math.sqrt(scale[i] + 1e-12);
        List<double[]> x = raw.stream().map(v -> standardise(v, mean, scale)).toList();

        Map<String, double[][]> weights = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            int[] y = targets.stream().mapToInt(t -> t.get(f.name())).toArray();
            weights.put(f.name(), fit(x, y, f.options().size(), seed));
        }
        return new EmbeddingBackend(modelId, embedder, mean, scale, weights);
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        double[] x = standardise(embedder.embed(request.state().render()), mean, scale);
        Map<String, RawScores> out = new LinkedHashMap<>();
        for (FieldSpec f : request.schema().fields()) {
            out.put(f.name(), new RawScores(LinearBackend.logSoftmax(logits(weights.get(f.name()), x))));
        }
        return out;
    }

    @Override
    public List<Map<String, RawScores>> scoreAll(List<ScoringRequest> requests) {
        embedder.embedAll(requests.stream().map(r -> r.state().render()).toList());
        return requests.stream().map(this::score).toList();
    }

    private static double[] standardise(float[] v, double[] mean, double[] scale) {
        double[] x = new double[v.length];
        for (int i = 0; i < v.length; i++) x[i] = (v[i] - mean[i]) * scale[i];
        return x;
    }

    /** @return weights per class, the last column of each row is the bias */
    static double[][] fit(List<double[]> x, int[] y, int classes, long seed) {
        int dim = x.getFirst().length;
        double[][] w = new double[classes][dim + 1], g2 = new double[classes][dim + 1];
        int[] order = new int[x.size()];
        Arrays.setAll(order, i -> i);
        Random random = new Random(seed);
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int n : order) {
                double[] xi = x.get(n);
                double[] p = LinearBackend.softmax(logits(w, xi));
                for (int k = 0; k < classes; k++) {
                    double err = p[k] - (k == y[n] ? 1 : 0);
                    double[] wk = w[k], gk = g2[k];
                    for (int i = 0; i < dim; i++) {
                        double g = err * xi[i] + L2 * wk[i];
                        gk[i] += g * g;
                        wk[i] -= RATE * g / (Math.sqrt(gk[i]) + 1e-8);
                    }
                    gk[dim] += err * err;
                    wk[dim] -= RATE * err / (Math.sqrt(gk[dim]) + 1e-8);
                }
            }
        }
        return w;
    }

    static double[] logits(double[][] w, double[] x) {
        double[] z = new double[w.length];
        for (int k = 0; k < w.length; k++) {
            double s = w[k][x.length];
            for (int i = 0; i < x.length; i++) s += w[k][i] * x[i];
            z[k] = s;
        }
        return z;
    }

    // ------------------------------------------------------------------------------------- embedder

    /**
     * Calls {@code POST <baseUrl>/embeddings} and remembers every vector on disk (one append-only file per model,
     * keyed by the SHA-256 of the text), so re-runs and training-size curves don't call the server again.
     */
    static final class Embedder {

        static final int BATCH = 32;

        private static final ObjectMapper JSON = new ObjectMapper();

        private final String baseUrl, model, apiKey;
        private final int concurrency;
        private final Path file;
        private final Map<String, float[]> cache = new ConcurrentHashMap<>();
        private final HttpClient http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(10)).build();

        Embedder(String baseUrl, String model, String apiKey, Path dir, int concurrency) {
            this.baseUrl = baseUrl.replaceAll("/+$", "");
            this.model = model;
            this.apiKey = apiKey;
            this.concurrency = concurrency;
            this.file = dir.resolve(model.replaceAll("[^A-Za-z0-9._-]+", "_") + ".bin");
            load();
        }

        String model() {
            return model;
        }

        float[] embed(String text) {
            float[] v = cache.get(key(text));
            if (v != null) return v;
            embedAll(List.of(text));
            return cache.get(key(text));
        }

        /** Fetches every text not cached yet, {@value #BATCH} per request, {@code concurrency} requests in flight. */
        void embedAll(List<String> texts) {
            List<String> todo = texts.stream().distinct().filter(t -> !cache.containsKey(key(t))).toList();
            if (todo.isEmpty()) return;
            if (todo.size() > BATCH) {
                System.err.printf(Locale.ROOT, "Embedding %d texts with %s, cache: %s%n", todo.size(), model, file);
            }
            long start = System.nanoTime();
            AtomicInteger done = new AtomicInteger();
            try (ExecutorService pool = Executors.newFixedThreadPool(concurrency)) {
                List<Future<?>> futures = new ArrayList<>();
                for (int i = 0; i < todo.size(); i += BATCH) {
                    List<String> batch = todo.subList(i, Math.min(todo.size(), i + BATCH));
                    futures.add(pool.submit(() -> {
                        List<float[]> vectors = request(batch);
                        Map<String, float[]> fresh = new LinkedHashMap<>();
                        for (int j = 0; j < batch.size(); j++) fresh.put(key(batch.get(j)), vectors.get(j));
                        append(fresh);
                        int n = done.addAndGet(batch.size());
                        if (todo.size() > BATCH && (n % (BATCH * 50) < BATCH || n == todo.size())) {
                            System.err.printf(Locale.ROOT, "  %d/%d texts, %.0f/s%n", n, todo.size(),
                                    n / ((System.nanoTime() - start) / 1e9));
                        }
                        return null;
                    }));
                }
                for (Future<?> f : futures) f.get();
            } catch (Exception e) {
                throw new IllegalStateException("Embedding failed: " + e.getMessage(), e);
            }
        }

        /** Wall time in ms of one uncached request per text (bypasses and does not fill the cache). */
        double[] timeSingle(List<String> texts) {
            double[] ms = new double[texts.size()];
            for (int i = 0; i < texts.size(); i++) {
                long t0 = System.nanoTime();
                request(List.of(texts.get(i)));
                ms[i] = (System.nanoTime() - t0) / 1e6;
            }
            return ms;
        }

        private List<float[]> request(List<String> batch) {
            String body;
            try {
                body = JSON.writeValueAsString(Map.of("model", model, "input", batch));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            HttpRequest.Builder rb = HttpRequest.newBuilder(URI.create(baseUrl + "/embeddings"))
                    .timeout(Duration.ofMinutes(2)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body));
            if (apiKey != null) rb.header("Authorization", "Bearer " + apiKey);
            for (int attempt = 0; ; attempt++) {
                try {
                    HttpResponse<String> r = http.send(rb.build(), HttpResponse.BodyHandlers.ofString());
                    if (r.statusCode() != 200) {
                        throw new IOException("POST /embeddings returned " + r.statusCode() + ": " + r.body());
                    }
                    float[][] out = new float[batch.size()][];
                    for (JsonNode d : JSON.readTree(r.body()).get("data")) {
                        JsonNode e = d.get("embedding");
                        float[] v = new float[e.size()];
                        for (int i = 0; i < v.length; i++) v[i] = (float) e.get(i).asDouble();
                        out[d.get("index").asInt()] = v;
                    }
                    return Arrays.asList(out);
                } catch (IOException e) {
                    if (attempt >= 2) throw new UncheckedIOException(e);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(e);
                }
            }
        }

        private void load() {
            if (!Files.exists(file)) return;
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
                byte[] hash = new byte[32];
                while (true) {
                    try {
                        in.readFully(hash);
                    } catch (EOFException end) {
                        return;
                    }
                    float[] v = new float[in.readInt()];
                    for (int i = 0; i < v.length; i++) v[i] = in.readFloat();
                    cache.put(HexFormat.of().formatHex(hash), v);
                }
            } catch (IOException e) {
                throw new UncheckedIOException("Cannot read embedding cache " + file, e);
            }
        }

        private synchronized void append(Map<String, float[]> fresh) {
            try {
                Files.createDirectories(file.getParent());
                try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND)))) {
                    for (Map.Entry<String, float[]> e : fresh.entrySet()) {
                        out.write(HexFormat.of().parseHex(e.getKey()));
                        out.writeInt(e.getValue().length);
                        for (float f : e.getValue()) out.writeFloat(f);
                    }
                }
            } catch (IOException e) {
                throw new UncheckedIOException("Cannot write embedding cache " + file, e);
            }
            cache.putAll(fresh);
        }

        private static String key(String text) {
            try {
                return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                        .digest(text.getBytes(StandardCharsets.UTF_8)));
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException(e);
            }
        }
    }
}
