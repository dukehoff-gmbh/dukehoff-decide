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

import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.classifier.LinearClassifier;
import com.dukehoff.decide.embedding.EmbeddingClassifier;
import com.dukehoff.decide.embedding.OpenAiEmbedder;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.showcase.Showcase.Config;
import com.dukehoff.decide.showcase.ShowcaseDataset.Example;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.spi.TaskSchema;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Time per decision with one example in flight and nothing cached (run with
 * {@code -Dshowcase.main=com.dukehoff.decide.showcase.Bench}). {@code -Dshowcase.backend=llm|linear|embedding}
 * and the dataset options as for {@link Showcase}; {@code -Dbench.n} timed examples after {@code -Dbench.warmup}.
 * A language model gets the fields of one example in parallel, as {@code OpenAiBackend} sends them; a trained
 * classifier is trained on {@code showcase.train} examples outside the timed ones (its scoring time does not depend
 * on that number) and, for embeddings, embeds each text with its own request. Prints one Markdown table row.
 */
public final class Bench {

    private Bench() {}

    public static void main(String[] args) throws Exception {
        Config cfg = Config.fromSystemProperties();
        run(ShowcaseDataset.byId(cfg.dataset()), cfg);
    }

    private static <T> void run(ShowcaseDataset<T> dataset, Config cfg) throws Exception {
        String engine = cfg.backend();
        int n = Integer.getInteger("bench.n", engine.equals("llm") ? 50 : 100);
        int warmup = Integer.getInteger("bench.warmup", engine.equals("linear") ? 500 : 5);
        List<Example<T>> all = dataset.load(cfg.cache().resolve("datasets"));
        List<Example<T>> examples = ShowcaseDataset.sample(all, warmup + n, cfg.seed(), cfg.language());
        TaskSchema schema = TaskSchema.of(dataset.type());

        ScoringBackend backend = switch (engine) {
            case "llm" -> OpenAiBackend.builder().baseUrl(cfg.baseUrl()).model(cfg.model())
                    .maxConcurrency(schema.fields().size()).timeout(Duration.ofMinutes(5)).build();
            case "linear" -> LinearClassifier.trainer().train(schema, training(all, examples.size(), cfg));
            case "embedding" -> embedding(schema, training(all, examples.size(), cfg), cfg);
            default -> throw new IllegalArgumentException("Unknown backend " + engine);
        };

        double[] ms = new double[n];
        for (int i = 0; i < warmup + n; i++) {
            ScoringRequest request = new ScoringRequest(examples.get(i).state(), schema);
            long start = System.nanoTime();
            backend.score(request);
            if (i >= warmup) ms[i - warmup] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(ms);
        String name = switch (engine) {
            case "llm" -> cfg.model();
            case "embedding" -> cfg.embeddingModel() + " + linear";
            default -> "linear";
        };
        System.out.printf(Locale.ROOT, "| %s %s | %s | %d | %d | %s | %s |%n", cfg.dataset(), cfg.language(), name,
                schema.fields().size(), n, format(ms[n / 2]), format(ms[(int) Math.ceil(0.95 * n) - 1]));
    }

    private static <T> List<Calibration.Labeled<T>> training(List<Example<T>> all, int taken, Config cfg) {
        return Showcase.labeled(ShowcaseDataset.rest(all, taken, cfg.train(), cfg.seed(), cfg.language(), 1).examples());
    }

    /** Trained on cached vectors, then loaded with an embedder without a cache, as in production. */
    private static <T> ScoringBackend embedding(TaskSchema schema, List<Calibration.Labeled<T>> training, Config cfg)
            throws Exception {
        OpenAiEmbedder.Builder b = OpenAiEmbedder.builder().baseUrl(cfg.embeddingUrl()).model(cfg.embeddingModel())
                .maxConcurrency(cfg.concurrency()).timeout(Duration.ofMinutes(2));
        if (cfg.apiKey() != null) b.apiKey(cfg.apiKey());
        OpenAiEmbedder embedder = b.build();
        EmbeddingClassifier trained = EmbeddingClassifier.trainer()
                .train(new CachingEmbedder(embedder, cfg.cache().resolve("embeddings")), schema, training);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        trained.save(bytes);
        return EmbeddingClassifier.load(new ByteArrayInputStream(bytes.toByteArray()), embedder);
    }

    private static String format(double ms) {
        return ms >= 1000 ? String.format(Locale.ROOT, "%.2f s", ms / 1000)
                : ms >= 1 ? String.format(Locale.ROOT, "%.0f ms", ms)
                : String.format(Locale.ROOT, "%.0f µs", ms * 1000);
    }
}
