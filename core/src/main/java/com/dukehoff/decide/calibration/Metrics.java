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

/** Standard classification and calibration metrics over (probabilities, target index) pairs. */
public final class Metrics {

    private Metrics() {}

    public static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    public static double accuracy(List<double[]> probs, int[] targets) {
        int correct = 0;
        for (int i = 0; i < targets.length; i++) if (argmax(probs.get(i)) == targets[i]) correct++;
        return targets.length == 0 ? 0 : (double) correct / targets.length;
    }

    /** Mean negative log-likelihood of the target option. Lower is better. */
    public static double nll(List<double[]> probs, int[] targets) {
        double sum = 0;
        for (int i = 0; i < targets.length; i++) sum -= Math.log(Math.max(probs.get(i)[targets[i]], 1e-12));
        return targets.length == 0 ? 0 : sum / targets.length;
    }

    /**
     * Expected calibration error with equal-width confidence bins: the weighted average gap
     * between "how sure the model said it was" and "how often it was right". 0 is perfect.
     */
    public static double ece(List<double[]> probs, int[] targets, int bins) {
        double[] confSum = new double[bins];
        double[] correctSum = new double[bins];
        int[] count = new int[bins];
        for (int i = 0; i < targets.length; i++) {
            double[] p = probs.get(i);
            int pred = argmax(p);
            double conf = p[pred];
            int b = Math.min(bins - 1, (int) (conf * bins));
            confSum[b] += conf;
            correctSum[b] += pred == targets[i] ? 1 : 0;
            count[b]++;
        }
        double ece = 0;
        for (int b = 0; b < bins; b++) {
            if (count[b] == 0) continue;
            ece += Math.abs(confSum[b] - correctSum[b]) / targets.length;
        }
        return ece;
    }

    public static double ece(List<double[]> probs, int[] targets) {
        return ece(probs, targets, 10);
    }
}
