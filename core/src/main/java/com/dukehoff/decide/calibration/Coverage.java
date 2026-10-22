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
import java.util.List;
import java.util.OptionalDouble;

/**
 * Coverage–accuracy curve: for each confidence threshold, how many examples a gate
 * ({@link com.dukehoff.decide.DecisionTask#requireConfidence(double)}) would decide automatically and how
 * many of those are right. The same rule as the gate: an example is decided if its confidence is
 * {@code >=} the threshold.
 *
 * @param points one per threshold in {@link #THRESHOLDS}, ascending
 */
public record Coverage(List<Point> points) {

    /** 0.50, 0.51, ..., 0.99. */
    public static final List<Double> THRESHOLDS;

    static {
        List<Double> t = new ArrayList<>();
        for (int i = 50; i <= 99; i++) t.add(i / 100.0);
        THRESHOLDS = List.copyOf(t);
    }

    /**
     * @param examples all examples
     * @param decided  examples with confidence {@code >= threshold}
     * @param correct  decided examples that were right
     */
    public record Point(double threshold, int examples, int decided, int correct) {

        /** Share decided automatically. */
        public double coverage() {
            return examples == 0 ? 0 : (double) decided / examples;
        }

        /** Accuracy on the decided share; NaN if nothing was decided. */
        public double accuracy() {
            return decided == 0 ? Double.NaN : (double) correct / decided;
        }
    }

    public Coverage {
        points = List.copyOf(points);
    }

    /**
     * @param confidence per example, as the gate sees it
     * @param correct    per example, whether the decided value is right
     */
    public static Coverage of(double[] confidence, boolean[] correct) {
        if (confidence.length != correct.length) throw new IllegalArgumentException("size mismatch");
        List<Point> points = new ArrayList<>(THRESHOLDS.size());
        for (double t : THRESHOLDS) {
            int decided = 0, right = 0;
            for (int i = 0; i < confidence.length; i++) {
                if (confidence[i] >= t) {
                    decided++;
                    if (correct[i]) right++;
                }
            }
            points.add(new Point(t, confidence.length, decided, right));
        }
        return new Coverage(points);
    }

    /** The point for {@code threshold}, which must be one of {@link #THRESHOLDS}. */
    public Point at(double threshold) {
        for (Point p : points) if (Math.abs(p.threshold() - threshold) < 1e-9) return p;
        throw new IllegalArgumentException("no point for threshold " + threshold + "; use one of 0.50, 0.51, ..., 0.99");
    }

    /**
     * The lowest threshold whose decided share reaches {@code targetAccuracy}, i.e. the one that decides the
     * most examples automatically at that accuracy. Empty if no threshold reaches it. Measured on the
     * evaluation data, so treat it as an estimate and prefer a holdout set.
     */
    public OptionalDouble thresholdFor(double targetAccuracy) {
        for (Point p : points) {
            if (p.decided() > 0 && p.accuracy() >= targetAccuracy) return OptionalDouble.of(p.threshold());
        }
        return OptionalDouble.empty();
    }
}
