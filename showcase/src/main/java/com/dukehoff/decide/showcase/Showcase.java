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

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration.CalibrationReport;
import com.dukehoff.decide.calibration.Calibration.FieldReport;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.classifier.LinearClassifier;
import com.dukehoff.decide.embedding.EmbeddingClassifier;
import com.dukehoff.decide.embedding.OpenAiEmbedder;
import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.calibration.Calibrator;
import com.dukehoff.decide.calibration.Confusion;
import com.dukehoff.decide.calibration.Coverage;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.showcase.ShowcaseDataset.Example;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.spi.TaskSchema;
import com.dukehoff.decide.testing.ScriptedBackend;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Runs a {@link ShowcaseDataset} end to end: fits temperature calibration on one split, then reports
 * on a disjoint test split: accuracy vs. a majority baseline, ECE/NLL before and after calibration,
 * how many examples can be decided automatically at a given confidence threshold (and how accurate
 * those are), and latency.
 *
 * <pre>
 * mvn -q install -DskipTests
 * mvn -q -pl showcase exec:java -Dshowcase.dataset=bitext -Dshowcase.model=qwen2.5-1.5b-instruct-q4_k_m
 * </pre>
 *
 * Settings (system properties): {@code showcase.dataset} (tickets, bitext, massive, cfpb; default tickets),
 * {@code showcase.baseUrl} (default http://localhost:8080/v1), {@code showcase.model}, {@code showcase.apiKey},
 * {@code showcase.calibration} (examples, default 300), {@code showcase.test} (default 300), {@code showcase.seed} (42),
 * {@code showcase.language} (all, en, de), {@code showcase.concurrency} (examples in flight, default 4),
 * {@code showcase.cache} (default ~/.cache/dukehoff-decide), {@code showcase.out} (default target/showcase/&lt;dataset&gt;),
 * {@code showcase.calibrator} (best, temperature; default best),
 * {@code showcase.backend} (llm, linear, embedding, combine; default llm), {@code showcase.combine} (see {@link Combined}), {@code showcase.train} (linear only: training examples,
 * disjoint from both splits, default 3000), {@code showcase.similarity} (linear only: drop training examples whose
 * character-trigram Jaccard similarity to a calibration or test example reaches this; default 1, exact duplicates),
 * {@code showcase.embeddingUrl} (default http://localhost:8081/v1), {@code showcase.embeddingModel} (default bge-m3).
 *
 * <p>{@code embedding} does the same on vectors from an embedding model ({@link EmbeddingClassifier}).
 * {@code linear} needs no model: it trains a {@link LinearClassifier} (TF-IDF + logistic regression) on examples the
 * calibration and test splits don't contain, so its report is directly comparable with an LLM run on the same seed.
 */
public final class Showcase {

    static final String WHOLE = "whole record";

    static final double[] THRESHOLDS = {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};

    record Config(String dataset, String baseUrl, String model, String apiKey, int calibration, int test, long seed,
                  String language, int concurrency, Path cache, Path out, Calibration.Method method,
                  String backend, int train, double similarity, String embeddingUrl, String embeddingModel) {

        static Config fromSystemProperties() {
            String backend = System.getProperty("showcase.backend", "llm");
            if (!List.of("llm", "linear", "embedding", "combine").contains(backend)) {
                throw new IllegalArgumentException("Unknown backend '" + backend + "', use llm, linear, embedding or combine");
            }
            boolean needsModel = backend.equals("llm")
                    || backend.equals("combine") && System.getProperty("showcase.combine", "").contains("llm");
            String model = System.getProperty("showcase.model", needsModel ? null : backend);
            if (model == null || model.isBlank()) {
                throw new IllegalArgumentException("Set -Dshowcase.model=<model name>, e.g. qwen2.5-1.5b-instruct-q4_k_m");
            }
            String dataset = System.getProperty("showcase.dataset", "tickets");
            if (dataset.equals("massive") && System.getProperty("showcase.language", "all").equals("all")) {
                throw new IllegalArgumentException("MASSIVE's German rows translate the English ones; "
                        + "set -Dshowcase.language=en or de");
            }
            return new Config(
                    dataset,
                    System.getProperty("showcase.baseUrl", "http://localhost:8080/v1"),
                    model,
                    System.getProperty("showcase.apiKey"),
                    Integer.getInteger("showcase.calibration", 300),
                    Integer.getInteger("showcase.test", 300),
                    Long.getLong("showcase.seed", 42),
                    System.getProperty("showcase.language", "all"),
                    Integer.getInteger("showcase.concurrency", 4),
                    Path.of(System.getProperty("showcase.cache", System.getProperty("user.home") + "/.cache/dukehoff-decide")),
                    Path.of(System.getProperty("showcase.out", "target/showcase/" + dataset
                            + (backend.equals("llm") ? "" : "-" + backend))),
                    Calibration.Method.valueOf(System.getProperty("showcase.calibrator", "best").toUpperCase(Locale.ROOT)),
                    backend,
                    Integer.getInteger("showcase.train", 3000),
                    Double.parseDouble(System.getProperty("showcase.similarity", "1")),
                    System.getProperty("showcase.embeddingUrl", "http://localhost:8081/v1"),
                    System.getProperty("showcase.embeddingModel", "bge-m3"));
        }
    }

    record Latency(int examples, double p50, double p95, double mean, int scoredNow, double perSecond) {}

    public static void main(String[] args) throws Exception {
        Config cfg = Config.fromSystemProperties();
        run(ShowcaseDataset.byId(cfg.dataset()), cfg);
    }

    static <T> void run(ShowcaseDataset<T> dataset, Config cfg) throws Exception {
        List<Example<T>> all = dataset.load(cfg.cache().resolve("datasets"));
        List<Example<T>> sample = ShowcaseDataset.sample(all, cfg.calibration() + cfg.test(), cfg.seed(), cfg.language());
        List<Example<T>> calibration = sample.subList(0, cfg.calibration());
        List<Example<T>> test = sample.subList(cfg.calibration(), sample.size());
        if (cfg.backend().equals("combine")) {
            Combined.run(dataset, cfg, all, calibration, test);
            return;
        }
        if (!cfg.backend().equals("llm")) {
            runTrained(dataset, cfg, all, calibration, test,
                    cfg.backend().equals("linear") ? linearTrainer() : embeddingTrainer(cfg));
            return;
        }

        OpenAiBackend.Builder b = OpenAiBackend.builder().baseUrl(cfg.baseUrl()).model(cfg.model())
                .maxConcurrency(cfg.concurrency() * dataset.fields().size());
        if (cfg.apiKey() != null) b.apiKey(cfg.apiKey());
        CachingBackend backend = new CachingBackend(b.build(), cfg.cache().resolve("scores"));

        DecisionTask<T> uncalibrated = Decider.builder().backend(backend).build().task(dataset.type());
        Latency latency = scoreAll(backend, uncalibrated.schema(), sample, cfg.concurrency());

        CalibrationReport fit = Calibration.fit(uncalibrated, labeled(calibration), labeled(test), cfg.method());
        CalibrationStore store = CalibrationStore.directory(cfg.out().resolve("calibration"));
        store.save(fit.profile());
        DecisionTask<T> calibrated = Decider.builder().backend(backend).calibration(store).build()
                .task(dataset.type());

        String model = "`" + cfg.model() + "` via `" + cfg.baseUrl() + "`";
        String report = report(dataset, cfg, model, calibration, test, calibrated, fit, llmLatency(dataset, cfg, latency));
        write(cfg, report);
    }

    /** Trains a backend without a prompt: features from the text, one logistic regression per field. */
    interface Trainer {
        /** Prefix of the model id, e.g. "linear-tfidf-lr". */
        String id();

        /** Markdown, for the report header. */
        String description();

        ScoringBackend train(String modelId, TaskSchema schema, List<State> states,
                             List<Map<String, Integer>> targets, long seed);

        /** Markdown paragraph about per-example cost beyond the classifier itself, may be empty. */
        default String cost(List<State> examples) {
            return "";
        }
    }

    static Trainer linearTrainer() {
        return new Trainer() {
            public String id() { return "linear-tfidf-lr"; }
            public String description() {
                return "TF-IDF (word 1-2-grams, char 3-5-grams) + logistic regression per field";
            }
            public ScoringBackend train(String modelId, TaskSchema schema, List<State> states,
                                        List<Map<String, Integer>> targets, long seed) {
                return LinearClassifier.trainer().seed(seed).modelId(modelId).train(schema, states, targets);
            }
        };
    }

    static Trainer embeddingTrainer(Config cfg) {
        OpenAiEmbedder.Builder b = OpenAiEmbedder.builder().baseUrl(cfg.embeddingUrl()).model(cfg.embeddingModel())
                .maxConcurrency(cfg.concurrency()).timeout(Duration.ofMinutes(2));
        if (cfg.apiKey() != null) b.apiKey(cfg.apiKey());
        CachingEmbedder embedder = new CachingEmbedder(b.build(), cfg.cache().resolve("embeddings"));
        return new Trainer() {
            public String id() { return "embed-" + embedder.modelId().replaceAll("[^A-Za-z0-9._-]+", "_") + "-lr"; }
            public String description() {
                return "embeddings from `" + embedder.modelId() + "` via `" + cfg.embeddingUrl()
                        + "` (frozen) + logistic regression per field";
            }
            public ScoringBackend train(String modelId, TaskSchema schema, List<State> states,
                                        List<Map<String, Integer>> targets, long seed) {
                return EmbeddingClassifier.trainer().seed(seed).modelId(modelId).train(embedder, schema, states, targets);
            }
            public String cost(List<State> examples) {
                double[] ms = embedder.timeSingle(examples.stream().limit(100).map(State::render).toList());
                Arrays.sort(ms);
                return String.format(Locale.ROOT, """
                        The table above assumes the embedding is already computed (they are cached). Computing one \
                        embedding (one request, one text, nothing else in flight), on %d examples: \
                        p50 %.0f ms, p95 %.0f ms, mean %.0f ms.
                        """, ms.length, percentile(ms, 0.5), percentile(ms, 0.95), Arrays.stream(ms).average().orElse(0));
            }
        };
    }

    /** Same splits and report as an LLM run, scored by a backend trained on examples outside both splits. */
    static <T> void runTrained(ShowcaseDataset<T> dataset, Config cfg, List<Example<T>> all,
                               List<Example<T>> calibration, List<Example<T>> test, Trainer trainer)
            throws Exception {
        ShowcaseDataset.Rest<T> rest = ShowcaseDataset.rest(all, calibration.size() + test.size(), cfg.train(),
                cfg.seed(), cfg.language(), cfg.similarity());
        List<Example<T>> training = rest.examples();
        String filter = cfg.similarity() >= 1 ? "exact duplicates of" : String.format(Locale.ROOT,
                "examples with character-trigram Jaccard similarity of at least %.2f to", cfg.similarity());
        System.err.printf(Locale.ROOT, "Dropped %d training candidates as %s the sample%n", rest.dropped(), filter);
        if (training.size() < cfg.train()) {
            System.err.printf(Locale.ROOT, "Only %d training examples available%n", training.size());
        }
        TaskSchema schema = Decider.builder().backend(new ScriptedBackend("schema", (s, f) -> ScriptedBackend.uniform(f)))
                .build().task(dataset.type()).schema();

        StringBuilder curve = new StringBuilder("\n## Training set size\n\n");
        curve.append("Accuracy on the test split (argmax, before calibration) by number of training examples. "
                + "Training examples never occur in the calibration or test split; " + rest.dropped()
                + " candidates were dropped as " + filter + " an example in either split.\n\n");
        List<String> names = new ArrayList<>(targets(dataset).keySet());
        curve.append(row(Stream.concat(Stream.of("training examples", "train time"),
                names.stream()).toArray(String[]::new)));
        curve.append("|---:|---:|").append("---:|".repeat(names.size())).append('\n');
        for (int n : new int[] {100, 300, 1000, 3000, 10000, 20000}) {
            if (n > training.size()) break;
            long start = System.nanoTime();
            ScoringBackend b = train(dataset, schema, training.subList(0, n), cfg, trainer);
            long ms = (System.nanoTime() - start) / 1_000_000;
            List<Decision<T>> d = Decider.builder().backend(b).build().task(dataset.type())
                    .decideAll(test.stream().map(Example::state).toList());
            List<T> expected = test.stream().map(Example::label).toList();
            List<String> cells = new ArrayList<>(List.of(Integer.toString(n), ms + " ms"));
            targets(dataset).values().forEach(get -> cells.add(pct(accuracy(d, expected, get))));
            curve.append(row(cells.toArray(String[]::new)));
            System.err.printf(Locale.ROOT, "  trained on %d examples in %d ms%n", n, ms);
        }

        long start = System.nanoTime();
        ScoringBackend backend = train(dataset, schema, training, cfg, trainer);
        long trainMillis = (System.nanoTime() - start) / 1_000_000;

        DecisionTask<T> uncalibrated = Decider.builder().backend(backend).build().task(dataset.type());
        CalibrationReport fit = Calibration.fit(uncalibrated, labeled(calibration), labeled(test), cfg.method());
        CalibrationStore store = CalibrationStore.directory(cfg.out().resolve("calibration"));
        store.save(fit.profile());
        DecisionTask<T> calibrated = Decider.builder().backend(backend).calibration(store).build()
                .task(dataset.type());

        List<State> sampleStates = Stream.concat(calibration.stream(), test.stream()).map(Example::state).toList();
        backend.scoreAll(sampleStates.stream().map(st -> new ScoringRequest(st, schema)).toList());
        double[] micros = new double[sampleStates.size()];
        for (int i = 0; i < micros.length; i++) {
            long t0 = System.nanoTime();
            backend.score(new ScoringRequest(sampleStates.get(i), schema));
            micros[i] = (System.nanoTime() - t0) / 1e3;
        }
        Arrays.sort(micros);
        String latency = String.format(Locale.ROOT, """

                ## Latency and cost

                Training on %d examples took %d ms (all fields, one thread, features already computed). \
                Scoring one example (all fields, one thread, warm JVM):

                | examples | p50 | p95 | mean |
                |---:|---:|---:|---:|
                | %d | %.0f µs | %.0f µs | %.0f µs |

                %s""", training.size(), trainMillis, micros.length, percentile(micros, 0.5), percentile(micros, 0.95),
                Arrays.stream(micros).average().orElse(0), trainer.cost(test.stream().map(Example::state).toList()));

        String model = String.format(Locale.ROOT, "`%s`: %s, trained on %d disjoint examples (%d dropped as %s "
                + "the splits)", backend.modelId(), trainer.description(), training.size(), rest.dropped(), filter);
        write(cfg, report(dataset, cfg, model, calibration, test, calibrated, fit, curve + latency));
    }

    static <T> ScoringBackend train(ShowcaseDataset<T> dataset, TaskSchema schema, List<Example<T>> training,
                                    Config cfg, Trainer trainer) {
        List<Map<String, Integer>> targets = training.stream().map(e -> {
            Map<String, Integer> t = new LinkedHashMap<>();
            for (FieldSpec f : schema.fields()) t.put(f.name(), optionIndex(dataset.fields().get(f.name()).apply(e.label())));
            return t;
        }).toList();
        return trainer.train(trainer.id() + "-" + training.size() + "-seed" + cfg.seed(), schema,
                training.stream().map(Example::state).toList(), targets, cfg.seed());
    }

    /** Option order of {@link FieldSpec}: yes before no, enum constants in ordinal order. */
    static int optionIndex(Object value) {
        return switch (value) {
            case Boolean b -> b ? 0 : 1;
            case Enum<?> e -> e.ordinal();
            default -> throw new IllegalArgumentException("unsupported field value " + value);
        };
    }

    private static void write(Config cfg, String report) throws Exception {
        Files.createDirectories(cfg.out());
        Path file = cfg.out().resolve("report.md");
        Files.writeString(file, report);
        System.out.println(report);
        System.err.println("Report written to " + file.toAbsolutePath());
    }

    /** Scores every example once (uncached ones hit the model), measuring per-example wall time. */
    static <T> Latency scoreAll(CachingBackend backend, TaskSchema schema, List<Example<T>> examples, int concurrency)
            throws Exception {
        List<ScoringRequest> requests = examples.stream().map(e -> new ScoringRequest(e.state(), schema)).toList();
        List<ScoringRequest> todo = requests.stream().filter(r -> backend.millis(r).isEmpty()).toList();
        long start = System.nanoTime();
        if (!todo.isEmpty()) {
            System.err.printf(Locale.ROOT, "Scoring %d of %d examples with %s (%d in parallel), cache: %s%n",
                    todo.size(), requests.size(), backend.modelId(), concurrency, backend.file());
            AtomicInteger done = new AtomicInteger();
            try (ExecutorService pool = Executors.newFixedThreadPool(concurrency)) {
                List<Future<?>> futures = new ArrayList<>();
                for (ScoringRequest r : todo) {
                    futures.add(pool.submit(() -> {
                        backend.score(r);
                        int n = done.incrementAndGet();
                        if (n % 25 == 0 || n == todo.size()) {
                            backend.save();
                            progress(n, todo.size(), System.nanoTime() - start);
                        }
                    }));
                }
                for (Future<?> f : futures) f.get();
            } finally {
                backend.save();
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;

        double[] ms = requests.stream().mapToDouble(r -> backend.millis(r).orElseThrow()).sorted().toArray();
        return new Latency(ms.length, percentile(ms, 0.5), percentile(ms, 0.95),
                Arrays.stream(ms).average().orElse(0), todo.size(),
                todo.isEmpty() ? Double.NaN : todo.size() / seconds);
    }

    private static void progress(int done, int total, long nanos) {
        double each = nanos / 1e9 / done;
        long eta = Math.round(each * (total - done));
        System.err.printf(Locale.ROOT, "  %d/%d examples, %.2f/s, ETA %s%n", done, total, 1 / each,
                Duration.ofSeconds(eta).toString().substring(2).toLowerCase(Locale.ROOT));
    }

    static double percentile(double[] sorted, double q) {
        if (sorted.length == 0) return Double.NaN;
        return sorted[(int) Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
    }

    static <T> List<Calibration.Labeled<T>> labeled(List<Example<T>> examples) {
        return examples.stream().map(Example::labeled).toList();
    }

    // ---------------------------------------------------------------------------------------- report

    static <T> String report(ShowcaseDataset<T> dataset, Config cfg, String model, List<Example<T>> calibration,
                             List<Example<T>> test, DecisionTask<T> calibrated,
                             CalibrationReport fit, String latency) {
        List<State> states = test.stream().map(Example::state).toList();
        List<T> expected = test.stream().map(Example::label).toList();
        List<Decision<T>> decisions = calibrated.decideAll(states);
        Map<String, Function<T, Object>> targets = targets(dataset);

        StringBuilder md = new StringBuilder();
        md.append("# dukehoff-decide showcase: ").append(dataset.title()).append("\n\n");
        md.append(String.format(Locale.ROOT, """
                - Model: %s
                - Data: %s, language: %s
                - Output type: `%s`
                - Split: %d examples for calibration, %d disjoint examples for testing (seed %d)
                - Calibrators allowed: %s
                - Date: %s

                """, model, dataset.citation(), cfg.language(), dataset.type().getSimpleName(),
                calibration.size(), test.size(), cfg.seed(),
                cfg.method() == Calibration.Method.BEST ? "temperature, or temperature + bias if better on the test split"
                        : "temperature only",
                LocalDate.now()));

        md.append("## Accuracy\n\n");
        md.append("Majority baseline: always predict the most frequent label of the calibration split.\n\n");
        md.append("| target | model | majority baseline |\n|---|---:|---:|\n");
        targets.forEach((name, get) -> md.append(row(name, pct(accuracy(decisions, expected, get)),
                pct(baseline(calibration, test, e -> get.apply(e.label()))))));

        md.append("\n## Calibration\n\n");
        md.append("Fitted on the calibration split, measured on the test split. "
                + "ECE: expected calibration error (10 bins), lower is better. "
                + "NLL (T only): what temperature scaling alone reaches.\n\n");
        md.append("| field | calibrator | ECE before | ECE after | NLL before | NLL after | NLL (T only) | temperature |\n"
                + "|---|---|---:|---:|---:|---:|---:|---:|\n");
        for (FieldReport f : fit.fields()) {
            md.append(row(f.field(), f.biased() ? "temperature + bias" : "temperature", f3(f.eceBefore()),
                    f3(f.eceAfter()), f3(f.nllBefore()), f3(f.nllAfter()), f3(f.nllTemperatureOnly()),
                    String.format(Locale.ROOT, "%.2f", f.temperature())));
        }
        for (FieldReport f : fit.fields()) {
            if (f.calibrator() instanceof Calibrator.Bias b) {
                md.append("\nBiases for ").append(f.field()).append(": ").append(b.biasByLabel().entrySet().stream()
                        .map(e -> e.getKey() + " " + String.format(Locale.ROOT, "%+.2f", e.getValue()))
                        .collect(Collectors.joining(", "))).append('\n');
            }
        }

        md.append("\n## Diagnostics\n\n");
        md.append("Where the errors go, on the test split. An option that collects most errors usually has a "
                + "description that matches too much.\n");
        for (FieldReport f : fit.fields()) {
            Confusion c = f.confusion();
            md.append("\n### ").append(f.field()).append("\n\n");
            for (String w : f.warnings()) md.append("> **Warning:** ").append(w).append("\n\n");
            if (c.labels().size() > 2) {
                md.append("| option | support | precision | recall |\n|---|---:|---:|---:|\n");
                for (int i = 0; i < c.labels().size(); i++) {
                    md.append(row(c.labels().get(i), Integer.toString(c.support(i)), pct(c.precision(i)),
                            pct(c.recall(i))));
                }
                md.append('\n');
            }
            List<Confusion.Cell> top = c.topConfusions(5);
            md.append(top.isEmpty() ? "No errors.\n" : "Top confusions (expected → predicted): "
                    + top.stream().map(x -> x.expected() + " → " + x.predicted() + " (" + x.count() + ")")
                            .collect(Collectors.joining(", ")) + "\n");
        }

        md.append("\n## Coverage and accuracy\n\n");
        md.append("Examples below the threshold go to a human. Coverage: share decided automatically. "
                + "Accuracy: on the automatically decided ones. The whole record is gated on its weakest field. "
                + "From the calibration report on the test split, with raw and with calibrated probabilities.\n");
        for (String target : targets.keySet()) {
            String name = target.equals(WHOLE) ? Calibration.WHOLE_RECORD : target;
            Coverage after = fit.coverage(name);
            Coverage before = target.equals(WHOLE) ? fit.wholeRecordBefore() : fit.fields().stream()
                    .filter(f -> f.field().equals(target)).findFirst().orElseThrow().coverageBefore();
            md.append("\n### ").append(target).append("\n\n");
            md.append("| threshold | coverage (raw) | accuracy (raw) | coverage (calibrated) | accuracy (calibrated) |\n"
                    + "|---:|---:|---:|---:|---:|\n");
            for (double t : THRESHOLDS) {
                Coverage.Point raw = before.at(t), cal = after.at(t);
                md.append(row(String.format(Locale.ROOT, "%.2f", t), pct(raw.coverage()), pct(raw.accuracy()),
                        pct(cal.coverage()), pct(cal.accuracy())));
            }
            md.append('\n');
            for (double goal : new double[] {0.9, 0.95}) {
                OptionalDouble t = fit.thresholdFor(name, goal);
                md.append(String.format(Locale.ROOT, "- %.0f%% accuracy: %s%n", 100 * goal, t.isEmpty()
                        ? "not reached at any threshold"
                        : String.format(Locale.ROOT, "threshold %.2f, decides %s automatically", t.getAsDouble(),
                                pct(after.at(t.getAsDouble()).coverage()))));
            }
        }

        md.append(latency);
        return md.toString();
    }

    static <T> String llmLatency(ShowcaseDataset<T> dataset, Config cfg, Latency latency) {
        int fields = dataset.fields().size();
        return "\n## Latency and cost\n\n" + String.format(Locale.ROOT, """
                One request per field, %d per example, each generating a single token. \
                Latency per example (fields scored concurrently, %d examples in flight, so this includes queueing):

                | examples | p50 | p95 | mean | throughput |
                |---:|---:|---:|---:|---:|
                | %d | %.0f ms | %.0f ms | %.0f ms | %s |

                Cost: local inference, hardware only. With a hosted API, cost is about %d requests per example \
                times the prompt size (a few hundred tokens), plus one output token each.
                """, fields, cfg.concurrency(), latency.examples(), latency.p50(), latency.p95(), latency.mean(),
                Double.isNaN(latency.perSecond()) ? "cached"
                        : String.format(Locale.ROOT, "%.2f examples/s", latency.perSecond()),
                fields);
    }

    /** Every field, then the whole record. */
    static <T> Map<String, Function<T, Object>> targets(ShowcaseDataset<T> dataset) {
        Map<String, Function<T, Object>> t = new LinkedHashMap<>(dataset.fields());
        t.put(WHOLE, v -> v);
        return t;
    }

    static <T> double accuracy(List<Decision<T>> decisions, List<T> expected, Function<T, Object> get) {
        int correct = 0;
        for (int i = 0; i < decisions.size(); i++) {
            if (get.apply(decisions.get(i).value()).equals(get.apply(expected.get(i)))) correct++;
        }
        return (double) correct / decisions.size();
    }

    static <T> double baseline(List<Example<T>> calibration, List<Example<T>> test, Function<Example<T>, Object> get) {
        Object majority = mode(calibration.stream().map(get).toList());
        return (double) test.stream().filter(e -> get.apply(e).equals(majority)).count() / test.size();
    }

    static Object mode(Collection<Object> values) {
        return values.stream().collect(Collectors.groupingBy(v -> v, LinkedHashMap::new, Collectors.counting()))
                .entrySet().stream().max(Map.Entry.comparingByValue()).orElseThrow().getKey();
    }

    private static String row(String... cells) {
        return "| " + String.join(" | ", cells) + " |\n";
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "-" : String.format(Locale.ROOT, "%.1f%%", 100 * v);
    }

    private static String f3(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }

    private Showcase() {}
}
