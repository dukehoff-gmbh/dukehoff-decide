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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Maps raw log-probs of one field to calibrated probabilities (summing to 1).
 * Sealed so that profiles can be persisted without serialising code.
 */
public sealed interface Calibrator {

    /** @param rawLogProbs one entry per option, in the field's option order */
    double[] apply(double[] rawLogProbs);

    /**
     * This calibrator for a field with the given option labels, in that order. Empty if it was fitted for
     * different options (an enum constant was renamed, added or removed), so the profile no longer applies.
     */
    default Optional<Calibrator> forOptions(List<String> labels) {
        return Optional.of(this);
    }

    /** Plain softmax: trusts the model's own probabilities. */
    record Identity() implements Calibrator {
        public double[] apply(double[] raw) { return softmax(raw, 1.0); }
    }

    /**
     * Temperature scaling (Guo et al. 2017): divides logits by {@code t} before the softmax.
     * t &gt; 1 softens an overconfident model, t &lt; 1 sharpens an underconfident one.
     * Never changes which option wins, only how sure the model claims to be.
     */
    record Temperature(double t) implements Calibrator {
        public Temperature {
            if (!(t > 0) || Double.isInfinite(t)) throw new IllegalArgumentException("temperature must be > 0: " + t);
        }

        public double[] apply(double[] raw) { return softmax(raw, t); }
    }

    /**
     * Temperature plus a bias per option ("bias scaling"): {@code softmax(logit / t + b)}. Corrects a model
     * that systematically favours or avoids an option, which a temperature alone cannot. Unlike
     * {@link Temperature}, it can change which option wins. Fitted by {@link BiasFit}.
     *
     * @param biasByLabel one bias per option label; iteration order is the order {@link #apply} expects
     */
    record Bias(double t, Map<String, Double> biasByLabel) implements Calibrator {
        public Bias {
            if (!(t > 0) || Double.isInfinite(t)) throw new IllegalArgumentException("temperature must be > 0: " + t);
            if (biasByLabel.isEmpty()) throw new IllegalArgumentException("no biases");
            biasByLabel.forEach((label, b) -> {
                if (b == null || !Double.isFinite(b)) throw new IllegalArgumentException("bias for " + label + ": " + b);
            });
            biasByLabel = Collections.unmodifiableMap(new LinkedHashMap<>(biasByLabel));
        }

        public double[] apply(double[] raw) {
            if (raw.length != biasByLabel.size()) {
                throw new IllegalArgumentException(raw.length + " scores for " + biasByLabel.size() + " biases");
            }
            double[] z = new double[raw.length];
            int i = 0;
            for (double b : biasByLabel.values()) {
                z[i] = raw[i] / t + b;
                i++;
            }
            return softmax(z, 1.0);
        }

        @Override
        public Optional<Calibrator> forOptions(List<String> labels) {
            if (labels.size() != biasByLabel.size() || !biasByLabel.keySet().containsAll(labels)) {
                return Optional.empty();
            }
            Map<String, Double> ordered = new LinkedHashMap<>();
            for (String l : labels) ordered.put(l, biasByLabel.get(l));
            return Optional.of(new Bias(t, ordered));
        }
    }

    static Calibrator identity() {
        return new Identity();
    }

    static Calibrator temperature(double t) {
        return new Temperature(t);
    }

    /** Numerically stable softmax of {@code x / t}. {@code -Infinity} entries get probability 0. */
    static double[] softmax(double[] x, double t) {
        if (x.length == 0) throw new IllegalArgumentException("no options");
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (Double.isNaN(v)) throw new IllegalArgumentException("NaN score");
            max = Math.max(max, v);
        }
        double[] out = new double[x.length];
        if (max == Double.NEGATIVE_INFINITY) {
            java.util.Arrays.fill(out, 1.0 / x.length);
            return out;
        }
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp((x[i] - max) / t);
            sum += out[i];
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }
}
