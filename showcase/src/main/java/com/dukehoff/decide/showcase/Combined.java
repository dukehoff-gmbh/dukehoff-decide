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

import com.dukehoff.decide.Cascade;
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Outcome;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.calibration.Calibration.CalibrationReport;
import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.classifier.LinearClassifier;
import com.dukehoff.decide.embedding.EmbeddingClassifier;
import com.dukehoff.decide.embedding.OpenAiEmbedder;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.showcase.Showcase.Config;
import com.dukehoff.decide.showcase.ShowcaseDataset.Example;
import com.dukehoff.decide.spi.Ensemble;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.TaskSchema;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Experiment ({@code -Dshowcase.backend=combine}): combines several backends on the same splits.
 * {@code showcase.combine} lists them in cascade order, e.g. {@code linear,embedding,llm} (default
 * {@code linear,embedding}); {@code llm} is {@code showcase.model} via {@code showcase.baseUrl}, usually from the
 * score cache. Each component is calibrated on its own. Then:
 * <ul>
 *   <li><b>ensemble</b>: a backend that averages the components' log-softmax scores, calibrated like any other;</li>
 *   <li><b>cascade</b>: the first component whose calibrated confidence reaches the threshold decides, otherwise
 *       the next one; if none does, the case goes to a human.</li>
 * </ul>
 * Trained classifiers are saved under {@code <cache>/models}, so re-runs only score.
 */
final class Combined {

    static final double[] THRESHOLDS = {0.8, 0.9, 0.95};

    private Combined() {}

    /** A backend from the list, calibrated on the calibration split, with its decisions on the test split. */
    record Component<T>(String name, ScoringBackend backend, CalibrationReport fit, DecisionTask<T> task,
                        List<Decision<T>> decisions) {}

    static <T> Seeds.Results run(ShowcaseDataset<T> dataset, Config cfg, List<Example<T>> all, List<Example<T>> calibration,
                        List<Example<T>> test) throws Exception {
        List<String> names = Arrays.stream(System.getProperty("showcase.combine", "linear,embedding").split(","))
                .map(String::strip).filter(s -> !s.isEmpty()).toList();
        ShowcaseDataset.Rest<T> rest = ShowcaseDataset.rest(all, calibration.size() + test.size(), cfg.train(),
                cfg.seed(), cfg.language(), cfg.similarity());
        List<Example<T>> training = rest.examples();
        TaskSchema schema = TaskSchema.of(dataset.type());
        String key = String.format(Locale.ROOT, "%s-%s-%d-%d-seed%d-train%d-sim%s", dataset.id(), cfg.language(),
                calibration.size(), test.size(), cfg.seed(), training.size(), cfg.similarity());

        List<Component<T>> components = new ArrayList<>();
        for (String name : names) {
            ScoringBackend backend = switch (name) {
                case "linear" -> linear(cfg, schema, training, key);
                case "embedding" -> embedding(cfg, schema, training, key);
                case "llm" -> llm(cfg, dataset);
                default -> throw new IllegalArgumentException("Unknown component '" + name + "', use linear, embedding or llm");
            };
            components.add(calibrate(name.equals("embedding") ? cfg.embeddingModel() : name.equals("llm") ? cfg.model() : name,
                    backend, dataset, cfg, calibration, test));
            System.err.println("  " + name + " calibrated");
        }
        Component<T> ensemble = calibrate("ensemble (mean log-probability)",
                Ensemble.of(components.stream().map(Component::backend).toList()), dataset, cfg, calibration, test);

        List<T> expected = test.stream().map(Example::label).toList();
        Seeds.Results results = new Seeds.Results();
        Map<String, java.util.function.Function<T, Object>> whole = Map.of(Showcase.WHOLE, v -> v);
        StringBuilder md = new StringBuilder();
        md.append("# dukehoff-decide showcase: combining backends, ").append(dataset.title()).append("\n\n");
        md.append(String.format(Locale.ROOT, """
                - Components (cascade order): %s
                - Data: %s, language: %s
                - Split: %d examples for calibration, %d disjoint examples for testing (seed %d); trained components \
                learn from %d other examples
                - Every component and the ensemble is calibrated on the calibration split (calibrators: %s)
                - Date: %s

                Whole-record decisions on the test split. Coverage: share decided automatically (calibrated confidence \
                at or above the threshold, minimum over fields). Accuracy: on the automatically decided ones.

                """, components.stream().map(c -> "`" + c.name() + "`").collect(Collectors.joining(" → ")),
                dataset.citation(), cfg.language(), calibration.size(), test.size(), cfg.seed(), training.size(),
                cfg.method(), LocalDate.now()));

        md.append("## Single backends and ensemble\n\n");
        md.append(header("backend", "accuracy"));
        List<Component<T>> rows = new ArrayList<>(components);
        rows.add(ensemble);
        for (Component<T> c : rows) {
            List<String> cells = new ArrayList<>(List.of("`" + c.name() + "`", pct(accuracy(c.decisions(), expected))));
            for (double t : THRESHOLDS) cells.addAll(gate(c.decisions(), expected, t));
            md.append(row(cells));
            results.putAll(c.name(), Seeds.metrics(c.decisions(), expected, whole));
        }

        md.append("\n## Cascade\n\n");
        md.append("The first component in the order above whose confidence reaches the threshold decides.\n\n");
        md.append(row(List.of("threshold", "coverage", "accuracy", "decided by")));
        md.append("|---:|---:|---:|---|\n");
        for (double t : THRESHOLDS) {
            Cascade<T> cascade = Cascade.of(components.stream().map(c -> c.task().requireConfidence(t)).toList());
            int decided = 0, correct = 0;
            int[] by = new int[components.size()];
            for (int i = 0; i < test.size(); i++) {
                Cascade.Step<T> step = cascade.step(test.get(i).state());
                if (step.outcome() instanceof Outcome.Confident<T>(Decision<T> d)) {
                    decided++;
                    by[step.stage()]++;
                    if (d.value().equals(expected.get(i))) correct++;
                }
            }
            Map<String, Double> m = new LinkedHashMap<>();
            Seeds.gate(m, t, decided, correct, test.size());
            for (int k = 0; k < by.length; k++) m.put("decided by " + components.get(k).name() + " " + f2(t), (double) by[k] / test.size());
            results.putAll("cascade", m);
            List<String> parts = new ArrayList<>();
            for (int k = 0; k < by.length; k++) parts.add(components.get(k).name() + " " + pct((double) by[k] / test.size()));
            md.append(row(List.of(f2(t), pct((double) decided / test.size()),
                    decided == 0 ? "-" : pct((double) correct / decided), String.join(", ", parts))));
        }

        md.append("\n## Agreement\n\n");
        int agree = 0, agreeCorrect = 0, anyCorrect = 0;
        for (int i = 0; i < test.size(); i++) {
            T first = components.getFirst().decisions().get(i).value();
            int n = i;
            if (components.stream().allMatch(c -> c.decisions().get(n).value().equals(first))) {
                agree++;
                if (first.equals(expected.get(i))) agreeCorrect++;
            }
            if (components.stream().anyMatch(c -> c.decisions().get(n).value().equals(expected.get(n)))) anyCorrect++;
        }
        md.append(String.format(Locale.ROOT, """
                All components pick the same answer on %s of the test cases, and that answer is right in %s of them. \
                At least one component is right on %s (an upper bound for any way of choosing between them).
                """, pct((double) agree / test.size()), agree == 0 ? "-" : pct((double) agreeCorrect / agree),
                pct((double) anyCorrect / test.size())));
        results.put("agreement", "all agree", (double) agree / test.size());
        results.put("agreement", "right when all agree", agree == 0 ? Double.NaN : (double) agreeCorrect / agree);
        results.put("agreement", "at least one right (oracle)", (double) anyCorrect / test.size());
        // paired: both measured on the same test cases of this seed, so the spread of the gain is much smaller
        Map<String, Double> first = results.rows.get(components.getFirst().name());
        for (String row : List.of(ensemble.name(), "cascade")) {
            for (double t : THRESHOLDS) {
                String at = String.format(Locale.ROOT, "@%.2f", t);
                for (String metric : List.of("coverage ", "accuracy ")) {
                    results.put("gain over " + components.getFirst().name(), metric + at + ", " + row,
                            results.rows.get(row).get(metric + at) - first.get(metric + at));
                }
            }
        }

        Files.createDirectories(cfg.out());
        Path file = cfg.out().resolve("report.md");
        Files.writeString(file, md);
        System.out.println(md);
        System.err.println("Report written to " + file.toAbsolutePath());
        return results;
    }

    // ------------------------------------------------------------------------------------- components

    private static <T> LinearClassifier linear(Config cfg, TaskSchema schema, List<Example<T>> training, String key) {
        Path file = cfg.cache().resolve("models").resolve("linear-" + key + ".model");
        if (Files.exists(file)) return LinearClassifier.load(file);
        LinearClassifier model = LinearClassifier.trainer().seed(cfg.seed())
                .train(schema, training.stream().map(Example::labeled).toList());
        save(file, () -> model.save(file));
        return model;
    }

    private static <T> EmbeddingClassifier embedding(Config cfg, TaskSchema schema, List<Example<T>> training, String key) {
        OpenAiEmbedder.Builder b = OpenAiEmbedder.builder().baseUrl(cfg.embeddingUrl()).model(cfg.embeddingModel())
                .maxConcurrency(cfg.concurrency()).timeout(Duration.ofMinutes(2));
        if (cfg.apiKey() != null) b.apiKey(cfg.apiKey());
        CachingEmbedder embedder = new CachingEmbedder(b.build(), cfg.cache().resolve("embeddings"));
        Path file = cfg.cache().resolve("models")
                .resolve("embedding-" + embedder.modelId().replaceAll("[^A-Za-z0-9._-]+", "_") + "-" + key + ".model");
        if (Files.exists(file)) return EmbeddingClassifier.load(file, embedder);
        EmbeddingClassifier model = EmbeddingClassifier.trainer().seed(cfg.seed())
                .train(embedder, schema, training.stream().map(Example::labeled).toList());
        save(file, () -> model.save(file));
        return model;
    }

    private static ScoringBackend llm(Config cfg, ShowcaseDataset<?> dataset) {
        OpenAiBackend.Builder b = OpenAiBackend.builder().baseUrl(cfg.baseUrl()).model(cfg.model())
                .maxConcurrency(cfg.concurrency() * dataset.fields().size());
        if (cfg.apiKey() != null) b.apiKey(cfg.apiKey());
        return new CachingBackend(b.build(), cfg.cache().resolve("scores"));
    }

    private static void save(Path file, Runnable write) {
        try {
            Files.createDirectories(file.getParent());
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
        write.run();
    }

    private static <T> Component<T> calibrate(String name, ScoringBackend backend, ShowcaseDataset<T> dataset,
                                              Config cfg, List<Example<T>> calibration, List<Example<T>> test) {
        DecisionTask<T> raw = Decider.builder().backend(backend).build().task(dataset.type());
        CalibrationReport fit = Calibration.fit(raw, Showcase.labeled(calibration), Showcase.labeled(test), cfg.method());
        CalibrationStore store = CalibrationStore.inMemory();
        store.save(fit.profile());
        DecisionTask<T> calibrated = Decider.builder().backend(backend).calibration(store).requireCalibration(true)
                .build().task(dataset.type());
        return new Component<>(name, backend, fit, calibrated,
                calibrated.decideAll(test.stream().map(Example::state).toList()));
    }

    // ------------------------------------------------------------------------------------- report

    private static <T> double accuracy(List<Decision<T>> decisions, List<T> expected) {
        int correct = 0;
        for (int i = 0; i < decisions.size(); i++) if (decisions.get(i).value().equals(expected.get(i))) correct++;
        return (double) correct / decisions.size();
    }

    private static <T> List<String> gate(List<Decision<T>> decisions, List<T> expected, double threshold) {
        int decided = 0, correct = 0;
        for (int i = 0; i < decisions.size(); i++) {
            if (decisions.get(i).confidence() >= threshold) {
                decided++;
                if (decisions.get(i).value().equals(expected.get(i))) correct++;
            }
        }
        return List.of(pct((double) decided / decisions.size()), decided == 0 ? "-" : pct((double) correct / decided));
    }

    private static String header(String first, String second) {
        List<String> cells = new ArrayList<>(List.of(first, second));
        for (double t : THRESHOLDS) {
            cells.add("coverage @" + f2(t));
            cells.add("accuracy @" + f2(t));
        }
        return row(cells) + "|---|---:|" + "---:|".repeat(2 * THRESHOLDS.length) + "\n";
    }

    private static String row(List<String> cells) {
        return "| " + String.join(" | ", cells) + " |\n";
    }

    private static String pct(double v) {
        return String.format(Locale.ROOT, "%.1f%%", 100 * v);
    }

    private static String f2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }
}
