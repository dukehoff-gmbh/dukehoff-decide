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

import java.util.List;

/**
 * Finds the temperature that minimises the negative log-likelihood on labelled data.
 * NLL is unimodal in log(t) for practical purposes, so a golden-section search is enough -
 * no gradient, no dependencies.
 */
public final class TemperatureFit {

    static final double MIN_T = 0.05;
    static final double MAX_T = 20.0;

    private TemperatureFit() {}

    public static double fit(List<double[]> rawLogProbs, int[] targets) {
        if (rawLogProbs.size() != targets.length) throw new IllegalArgumentException("size mismatch");
        if (targets.length == 0) return 1.0;
        double a = Math.log(MIN_T);
        double b = Math.log(MAX_T);
        final double phi = (Math.sqrt(5) - 1) / 2;
        double c = b - phi * (b - a);
        double d = a + phi * (b - a);
        double fc = nll(rawLogProbs, targets, Math.exp(c));
        double fd = nll(rawLogProbs, targets, Math.exp(d));
        for (int i = 0; i < 80 && b - a > 1e-6; i++) {
            if (fc < fd) {
                b = d; d = c; fd = fc;
                c = b - phi * (b - a);
                fc = nll(rawLogProbs, targets, Math.exp(c));
            } else {
                a = c; c = d; fc = fd;
                d = a + phi * (b - a);
                fd = nll(rawLogProbs, targets, Math.exp(d));
            }
        }
        return Math.exp((a + b) / 2);
    }

    static double nll(List<double[]> raw, int[] targets, double t) {
        double sum = 0;
        for (int i = 0; i < targets.length; i++) {
            double[] p = Calibrator.softmax(raw.get(i), t);
            sum -= Math.log(Math.max(p[targets[i]], 1e-12));
        }
        return sum / targets.length;
    }
}
