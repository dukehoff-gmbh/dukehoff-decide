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

import com.dukehoff.decide.Decision;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * Repeats a run over several seeds ({@code -Dshowcase.seeds=42,43,44,45,46}): each seed draws its own calibration,
 * test and training examples. The summary gives mean, standard deviation and range of the headline numbers.
 */
final class Seeds {

    static final double[] THRESHOLDS = {0.8, 0.9, 0.95};

    /** Row (backend or variant) → metric → value, both in insertion order. */
    static final class Results {
        final Map<String, Map<String, Double>> rows = new LinkedHashMap<>();

        void put(String row, String metric, double value) {
            rows.computeIfAbsent(row, r -> new LinkedHashMap<>()).put(metric, value);
        }

        void putAll(String row, Map<String, Double> metrics) {
            metrics.forEach((m, v) -> put(row, m, v));
        }
    }

    private Seeds() {}

    /** Accuracy per field and for the whole record, then whole-record coverage and accuracy per threshold. */
    static <T> Map<String, Double> metrics(List<Decision<T>> decisions, List<T> expected,
                                           Map<String, Function<T, Object>> targets) {
        Map<String, Double> m = new LinkedHashMap<>();
        targets.forEach((name, get) -> m.put("accuracy " + name, Showcase.accuracy(decisions, expected, get)));
        for (double t : THRESHOLDS) {
            int decided = 0, correct = 0;
            for (int i = 0; i < decisions.size(); i++) {
                if (decisions.get(i).confidence() >= t) {
                    decided++;
                    if (decisions.get(i).value().equals(expected.get(i))) correct++;
                }
            }
            gate(m, t, decided, correct, decisions.size());
        }
        return m;
    }

    /** Coverage and accuracy at one threshold; accuracy is left out when nothing was decided. */
    static void gate(Map<String, Double> m, double threshold, int decided, int correct, int examples) {
        String t = String.format(Locale.ROOT, "@%.2f", threshold);
        m.put("coverage " + t, (double) decided / examples);
        m.put("accuracy " + t, decided == 0 ? Double.NaN : (double) correct / decided);
    }

    static String report(String title, String setup, List<Long> seeds, List<Results> runs) {
        StringBuilder md = new StringBuilder("# ").append(title).append(" over ").append(seeds.size())
                .append(" seeds\n\n").append(setup).append("\n\n");
        md.append("Seeds: ").append(seeds).append(". Each seed draws its own calibration, test and training examples. "
                + "Cells: mean ± standard deviation (min–max), in percent. Accuracy at a threshold is averaged over "
                + "the seeds where anything was decided.\n");
        Set<String> rows = new LinkedHashSet<>();
        runs.forEach(r -> rows.addAll(r.rows.keySet()));
        for (String row : rows) {
            Set<String> metrics = new LinkedHashSet<>();
            runs.forEach(r -> metrics.addAll(r.rows.getOrDefault(row, Map.of()).keySet()));
            md.append("\n### ").append(row).append("\n\n| metric | mean ± sd | range | seeds |\n|---|---:|---:|---:|\n");
            for (String metric : metrics) {
                List<Double> values = new ArrayList<>();
                for (Results r : runs) {
                    Double v = r.rows.getOrDefault(row, Map.of()).get(metric);
                    if (v != null && !v.isNaN()) values.add(v);
                }
                md.append("| ").append(metric).append(" | ").append(meanSd(values)).append(" | ").append(range(values))
                        .append(" | ").append(values.size()).append(" |\n");
            }
        }
        return md.toString();
    }

    static String meanSd(List<Double> v) {
        if (v.isEmpty()) return "-";
        double mean = v.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        double ss = v.stream().mapToDouble(x -> (x - mean) * (x - mean)).sum();
        double sd = v.size() > 1 ? Math.sqrt(ss / (v.size() - 1)) : 0;
        return String.format(Locale.ROOT, "%.1f ± %.1f", 100 * mean, 100 * sd);
    }

    static String range(List<Double> v) {
        if (v.isEmpty()) return "-";
        return String.format(Locale.ROOT, "%.1f to %.1f", 100 * v.stream().mapToDouble(Double::doubleValue).min().orElseThrow(),
                100 * v.stream().mapToDouble(Double::doubleValue).max().orElseThrow());
    }
}
