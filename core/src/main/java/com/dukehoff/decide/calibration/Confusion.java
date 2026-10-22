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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Confusion matrix of one field: which option the model picked for which expected option.
 * The main lever for a field with poor accuracy is usually its option descriptions, and this
 * shows which ones: an option that collects most of the errors (an "attractor") typically has a
 * description that matches too much.
 *
 * @param labels option labels, in {@link com.dukehoff.decide.spi.FieldSpec#options()} order
 * @param counts {@code counts[expected][predicted]}
 */
public record Confusion(List<String> labels, int[][] counts) {

    /** Errors of one kind: {@code count} examples of {@code expected} were predicted as {@code predicted}. */
    public record Cell(String expected, String predicted, int count) {}

    /**
     * An option that receives a disproportionate share of the errors.
     *
     * @param share fraction of all errors of the field that were predicted as this option
     */
    public record Attractor(String label, int errors, double share) {}

    public Confusion {
        labels = List.copyOf(labels);
        int[][] copy = new int[labels.size()][];
        for (int i = 0; i < copy.length; i++) {
            if (counts[i].length != labels.size()) throw new IllegalArgumentException("counts must be square");
            copy[i] = counts[i].clone();
        }
        counts = copy;
    }

    /** Predictions are the argmax of {@code probs}. */
    public static Confusion of(List<String> labels, List<double[]> probs, int[] targets) {
        int[][] counts = new int[labels.size()][labels.size()];
        for (int i = 0; i < targets.length; i++) counts[targets[i]][Metrics.argmax(probs.get(i))]++;
        return new Confusion(labels, counts);
    }

    @Override
    public int[][] counts() {
        int[][] copy = new int[counts.length][];
        for (int i = 0; i < copy.length; i++) copy[i] = counts[i].clone();
        return copy;
    }

    public int examples() {
        int n = 0;
        for (int[] row : counts) for (int c : row) n += c;
        return n;
    }

    public int errors() {
        int n = examples();
        for (int i = 0; i < counts.length; i++) n -= counts[i][i];
        return n;
    }

    /** Examples whose expected option is {@code option}. */
    public int support(int option) {
        int n = 0;
        for (int c : counts[option]) n += c;
        return n;
    }

    /** Of the examples predicted as {@code option}, the share that was right. NaN if never predicted. */
    public double precision(int option) {
        int predicted = 0;
        for (int[] row : counts) predicted += row[option];
        return predicted == 0 ? Double.NaN : (double) counts[option][option] / predicted;
    }

    /** Of the examples expected to be {@code option}, the share predicted correctly. NaN if none. */
    public double recall(int option) {
        int support = support(option);
        return support == 0 ? Double.NaN : (double) counts[option][option] / support;
    }

    /** The {@code n} most frequent errors, largest first. */
    public List<Cell> topConfusions(int n) {
        List<Cell> cells = new ArrayList<>();
        for (int e = 0; e < counts.length; e++) {
            for (int p = 0; p < counts.length; p++) {
                if (e != p && counts[e][p] > 0) cells.add(new Cell(labels.get(e), labels.get(p), counts[e][p]));
            }
        }
        cells.sort(Comparator.comparingInt(Cell::count).reversed());
        return List.copyOf(cells.subList(0, Math.min(n, cells.size())));
    }

    /**
     * The option that received the most errors, if it received at least 5 of them, at least 40% of
     * all errors, and at least 1.5 times its share under evenly spread errors ({@code 1 / (options - 1)}).
     * Needs 3 or more options; with two, every error lands on the other option by definition.
     */
    public Optional<Attractor> attractor() {
        int k = labels.size();
        int errors = errors();
        if (k < 3 || errors == 0) return Optional.empty();
        int best = -1, bestCount = -1;
        for (int p = 0; p < k; p++) {
            int received = 0;
            for (int e = 0; e < k; e++) if (e != p) received += counts[e][p];
            if (received > bestCount) { best = p; bestCount = received; }
        }
        double share = (double) bestCount / errors;
        if (bestCount >= 5 && share >= 0.4 && share >= 1.5 / (k - 1)) {
            return Optional.of(new Attractor(labels.get(best), bestCount, share));
        }
        return Optional.empty();
    }
}
