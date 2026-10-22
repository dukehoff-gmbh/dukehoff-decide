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
package com.dukehoff.decide.calibration;

import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.State;
import com.dukehoff.decide.internal.Binder;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.TaskSchema;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Fitting calibration from labelled examples. Runs offline (test, CLI, build step);
 * the resulting profile is stored and picked up by the Decider at runtime.
 *
 * <pre>
 * var report = Calibration.fit(triage, train, holdout);
 * System.out.println(report);   // accuracy, ECE, coverage per threshold, confusions, warnings
 * store.save(report.profile());
 * double t = report.thresholdFor("priority", 0.95).orElse(1.0);   // gate that reaches 95% accuracy
 * </pre>
 *
 * Historical decisions make good labels: tickets and the team that finally handled them,
 * invoices and how they were booked.
 */
public final class Calibration {

    private Calibration() {}

    public record Labeled<T>(State state, T expected) {}

    /** Which calibrators {@link #fit} may choose from. */
    public enum Method {
        /** Temperature only. Never changes which option wins. */
        TEMPERATURE,
        /**
         * Temperature, or temperature plus a bias per option if that has a lower NLL on the holdout set.
         * Without a holdout set, temperature.
         */
        BEST
    }

    /**
     * One calibrator per field, valid only for (taskId, modelId).
     *
     * @param promptHashByField {@link LabelPrompt#fingerprint} of each field at fitting time; a field
     *                          without one (hand-written profiles) is not checked
     */
    public record Profile(String taskId, String modelId, Map<String, Calibrator> byField,
                          Map<String, String> promptHashByField) {
        public Profile {
            byField = Map.copyOf(byField);
            promptHashByField = Map.copyOf(promptHashByField);
        }

        /** Without prompt fingerprints. */
        public Profile(String taskId, String modelId, Map<String, Calibrator> byField) {
            this(taskId, modelId, byField, Map.of());
        }
    }

    /**
     * Metrics for one field, measured on the evaluation set (holdout if given, else training data).
     *
     * @param accuracy            after calibration; with a bias calibrator it can differ from the raw model's
     * @param temperature         of the chosen calibrator
     * @param calibrator          the chosen calibrator, stored in the profile
     * @param nllTemperatureOnly  evaluation NLL of temperature scaling alone, for comparison with {@code nllAfter}
     * @param confusion           which option was predicted for which expected option
     * @param coverageBefore      coverage–accuracy curve with the model's raw probabilities
     * @param coverageAfter       coverage–accuracy curve after calibration; what a gate on this field will do
     */
    public record FieldReport(String field, int examples, double accuracy,
                              double eceBefore, double eceAfter,
                              double nllBefore, double nllAfter, double temperature,
                              Calibrator calibrator, double nllTemperatureOnly,
                              Confusion confusion, Coverage coverageBefore, Coverage coverageAfter) {

        /** True if a per-option bias was chosen over temperature scaling. */
        public boolean biased() {
            return calibrator instanceof Calibrator.Bias;
        }

        /**
         * True if the fitted temperature hit the search range. At the upper end the model's scores
         * carry almost no information about this field; at the lower end they are far too timid.
         */
        public boolean temperatureAtLimit() {
            return temperature >= TemperatureFit.MAX_T * 0.99 || temperature <= TemperatureFit.MIN_T * 1.01;
        }

        /** Human-readable hints: attractor options and temperatures at the limit. Empty if nothing stands out. */
        public List<String> warnings() {
            List<String> w = new ArrayList<>();
            confusion.attractor().ifPresent(a -> w.add(String.format(Locale.ROOT,
                    "%s: option %s receives %d of %d errors (%.0f%%); its description probably matches too much",
                    field, a.label(), a.errors(), confusion.errors(), 100 * a.share())));
            if (temperatureAtLimit()) {
                w.add(String.format(Locale.ROOT, temperature > 1
                        ? "%s: temperature %.2f is at the limit; the model's scores barely relate to the labels"
                        : "%s: temperature %.2f is at the limit; the model is far less sure than it should be",
                        field, temperature));
            }
            return w;
        }
    }

    /** Target name for {@link CalibrationReport#coverage(String)}: all fields gated together. */
    public static final String WHOLE_RECORD = "(whole record)";

    /**
     * @param evaluatedOnHoldout false means the numbers are in-sample and optimistic
     * @param wholeRecordBefore  coverage–accuracy curve of {@code requireConfidence(threshold)} on all fields,
     *                           i.e. gated on the weakest field, with raw probabilities; correct means every
     *                           field is right
     * @param wholeRecordAfter   the same after calibration
     */
    public record CalibrationReport(Profile profile, List<FieldReport> fields, boolean evaluatedOnHoldout,
                                    Coverage wholeRecordBefore, Coverage wholeRecordAfter) {

        /** Thresholds shown by {@link #toString()}. */
        static final double[] SHOWN = {0.5, 0.7, 0.8, 0.9, 0.95, 0.99};

        /** The calibrated curve of a field, or of all fields with {@link #WHOLE_RECORD}. */
        public Coverage coverage(String target) {
            if (WHOLE_RECORD.equals(target)) return wholeRecordAfter;
            return fields.stream().filter(f -> f.field().equals(target)).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("No field '" + target + "' in " + profile.taskId()))
                    .coverageAfter();
        }

        /**
         * The lowest threshold at which gating {@code target} (a field, or {@link #WHOLE_RECORD}) reached
         * {@code targetAccuracy} on the evaluation data. Empty if none did.
         */
        public OptionalDouble thresholdFor(String target, double targetAccuracy) {
            return coverage(target).thresholdFor(targetAccuracy);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Calibration of ").append(profile.taskId()).append(" on ").append(profile.modelId())
              .append(evaluatedOnHoldout ? " (holdout)" : " (in-sample, optimistic)").append('\n');
            sb.append(String.format(Locale.ROOT, "%-16s %6s %8s %10s %10s %6s %s%n",
                    "field", "n", "accuracy", "ECE before", "ECE after", "T", "calibrator"));
            for (FieldReport f : fields) {
                sb.append(String.format(Locale.ROOT, "%-16s %6d %7.1f%% %10.3f %10.3f %6.2f %s%n",
                        f.field(), f.examples(), 100 * f.accuracy(), f.eceBefore(), f.eceAfter(), f.temperature(),
                        f.biased() ? "temperature + bias" : "temperature"));
            }
            for (FieldReport f : fields) {
                if (!(f.calibrator() instanceof Calibrator.Bias b)) continue;
                sb.append(String.format(Locale.ROOT, "%s: bias chosen, NLL %.3f vs %.3f with temperature only; biases:",
                        f.field(), f.nllAfter(), f.nllTemperatureOnly()));
                b.biasByLabel().entrySet().stream()
                        .sorted((x, y) -> Double.compare(Math.abs(y.getValue()), Math.abs(x.getValue())))
                        .limit(5)
                        .forEach(e -> sb.append(String.format(Locale.ROOT, " %s %+.2f", e.getKey(), e.getValue())));
                sb.append('\n');
            }
            sb.append("decided automatically after calibration (share / accuracy of those), by threshold\n");
            sb.append(String.format(Locale.ROOT, "%-16s", "field"));
            for (double t : SHOWN) sb.append(String.format(Locale.ROOT, " %13.2f", t));
            sb.append('\n');
            List<String> targets = new ArrayList<>(fields.stream().map(FieldReport::field).toList());
            if (fields.size() > 1) targets.add(WHOLE_RECORD);
            for (String target : targets) {
                sb.append(String.format(Locale.ROOT, "%-16s", target));
                for (double t : SHOWN) {
                    Coverage.Point p = coverage(target).at(t);
                    sb.append(p.decided() == 0 ? String.format(Locale.ROOT, " %13s", "0%")
                            : String.format(Locale.ROOT, " %5.1f%%/%5.1f%%", 100 * p.coverage(), 100 * p.accuracy()));
                }
                sb.append('\n');
            }
            for (FieldReport f : fields) {
                List<Confusion.Cell> top = f.confusion().topConfusions(3);
                if (top.isEmpty()) continue;
                sb.append(f.field()).append(" top confusions (expected -> predicted):");
                for (int i = 0; i < top.size(); i++) {
                    Confusion.Cell c = top.get(i);
                    sb.append(i == 0 ? " " : ", ").append(c.expected()).append(" -> ").append(c.predicted())
                      .append(' ').append(c.count());
                }
                sb.append('\n');
            }
            for (FieldReport f : fields) {
                for (String w : f.warnings()) sb.append("! ").append(w).append('\n');
            }
            return sb.toString();
        }
    }

    /** Fits and evaluates on the same data. Fine for a first look; use the holdout variant for real numbers. */
    public static <T> CalibrationReport fit(DecisionTask<T> task, List<Labeled<T>> examples) {
        return fit(task, examples, List.of());
    }

    /** {@link Method#BEST}: temperature, or temperature plus bias where that is better on the holdout set. */
    public static <T> CalibrationReport fit(DecisionTask<T> task, List<Labeled<T>> train, List<Labeled<T>> holdout) {
        return fit(task, train, holdout, Method.BEST);
    }

    /**
     * Fits calibrators on {@code train} and reports metrics on {@code holdout} (or on train if empty). With
     * {@link Method#BEST}, the holdout set also decides between the calibrators, so its numbers are slightly
     * optimistic for the chosen one.
     */
    public static <T> CalibrationReport fit(DecisionTask<T> task, List<Labeled<T>> train, List<Labeled<T>> holdout,
                                            Method method) {
        if (train.isEmpty()) throw new IllegalArgumentException("no training examples");
        TaskSchema schema = task.schema();
        Scored trainScored = score(task, schema, train);
        Scored evalScored = holdout.isEmpty() ? trainScored : score(task, schema, holdout);

        Map<String, Calibrator> calibrators = new LinkedHashMap<>();
        List<FieldReport> reports = new ArrayList<>();
        int n = evalScored.size();
        Gate wholeBefore = new Gate(n), wholeAfter = new Gate(n);
        for (FieldSpec f : schema.fields()) {
            List<String> labels = f.options().stream().map(FieldSpec.OptionSpec::label).toList();
            List<double[]> trainRaw = trainScored.raw.get(f.name());
            int[] trainTargets = trainScored.targets(f.name());
            List<double[]> raw = evalScored.raw.get(f.name());
            int[] targets = evalScored.targets(f.name());

            Calibrator temperature = Calibrator.temperature(TemperatureFit.fit(trainRaw, trainTargets));
            double nllTemperature = Metrics.nll(raw.stream().map(temperature::apply).toList(), targets);
            Calibrator cal = temperature;
            if (method == Method.BEST && !holdout.isEmpty()) {
                Optional<Calibrator.Bias> bias = BiasFit.fit(trainRaw, trainTargets, labels);
                if (bias.isPresent() && Metrics.nll(raw.stream().map(bias.get()::apply).toList(), targets) < nllTemperature) {
                    cal = bias.get();
                }
            }
            calibrators.put(f.name(), cal);
            double t = switch (cal) {
                case Calibrator.Temperature x -> x.t();
                case Calibrator.Bias x -> x.t();
                case Calibrator.Identity x -> 1.0;
            };

            List<double[]> before = raw.stream().map(r -> Calibrator.softmax(r, 1.0)).toList();
            List<double[]> after = raw.stream().map(cal::apply).toList();
            Gate fieldBefore = new Gate(n), fieldAfter = new Gate(n);
            fieldBefore.add(f, before, targets);
            fieldAfter.add(f, after, targets);
            wholeBefore.add(f, before, targets);
            wholeAfter.add(f, after, targets);
            reports.add(new FieldReport(f.name(), targets.length, Metrics.accuracy(after, targets),
                    Metrics.ece(before, targets), Metrics.ece(after, targets),
                    Metrics.nll(before, targets), Metrics.nll(after, targets), t, cal, nllTemperature,
                    Confusion.of(labels, after, targets), fieldBefore.coverage(), fieldAfter.coverage()));
        }
        Map<String, String> promptHashes = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) promptHashes.put(f.name(), LabelPrompt.fingerprint(schema, f));
        Profile profile = new Profile(schema.taskId(), task.modelId(), calibrators, promptHashes);
        return new CalibrationReport(profile, reports, !holdout.isEmpty(), wholeBefore.coverage(), wholeAfter.coverage());
    }

    /**
     * What a gate sees per example: the minimum confidence over the fields added so far (as in
     * {@code Decision.confidence()}, computed by the same binding code) and whether all of them are right
     * (the most likely option is the expected one).
     */
    private static final class Gate {
        final double[] confidence;
        final boolean[] correct;

        Gate(int n) {
            confidence = new double[n];
            correct = new boolean[n];
            java.util.Arrays.fill(confidence, 1.0);
            java.util.Arrays.fill(correct, true);
        }

        void add(FieldSpec f, List<double[]> probs, int[] targets) {
            for (int i = 0; i < targets.length; i++) {
                double[] p = probs.get(i);
                confidence[i] = Math.min(confidence[i], Binder.result(f, p).confidence());
                correct[i] &= Metrics.argmax(p) == targets[i];
            }
        }

        Coverage coverage() {
            return Coverage.of(confidence, correct);
        }
    }

    private record Scored(Map<String, List<double[]>> raw, Map<String, List<Integer>> targetLists) {
        int size() {
            return targetLists.values().iterator().next().size();
        }

        int[] targets(String field) {
            return targetLists.get(field).stream().mapToInt(Integer::intValue).toArray();
        }
    }

    private static <T> Scored score(DecisionTask<T> task, TaskSchema schema, List<Labeled<T>> examples) {
        List<Map<String, RawScores>> all = task.rawScores(examples.stream().map(Labeled::state).toList());
        Map<String, List<double[]>> raw = new LinkedHashMap<>();
        Map<String, List<Integer>> targets = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            raw.put(f.name(), new ArrayList<>());
            targets.put(f.name(), new ArrayList<>());
        }
        for (int i = 0; i < examples.size(); i++) {
            Map<String, Integer> idx = Binder.targetIndices(schema, examples.get(i).expected());
            for (FieldSpec f : schema.fields()) {
                raw.get(f.name()).add(all.get(i).get(f.name()).logProbs());
                targets.get(f.name()).add(idx.get(f.name()));
            }
        }
        return new Scored(raw, targets);
    }
}
