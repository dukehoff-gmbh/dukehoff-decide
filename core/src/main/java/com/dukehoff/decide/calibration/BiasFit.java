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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Fits {@link Calibrator.Bias}: minimises the mean negative log-likelihood of
 * {@code softmax(s * logit + b)} plus {@code lambda / 2 * |b|^2}, with {@code s = 1 / t}.
 *
 * <p>In {@code (s, b)} this is multinomial logistic regression with two kinds of features (the raw logit and
 * an indicator per option), so the loss is convex and Newton's method converges in a few steps. There is no
 * minimum number of examples per option: the L2 penalty keeps biases of rare options near zero, and
 * {@link Calibration#fit} only keeps the result if it beats temperature scaling on the holdout set. The penalty
 * on all biases also removes the softmax's shift ambiguity; at the optimum the biases sum to zero.
 */
public final class BiasFit {

    /** L2 strength on the biases, relative to the mean NLL. */
    public static final double DEFAULT_LAMBDA = 0.01;

    private BiasFit() {}

    /** With {@link #DEFAULT_LAMBDA}. */
    public static Optional<Calibrator.Bias> fit(List<double[]> rawLogProbs, int[] targets, List<String> labels) {
        return fit(rawLogProbs, targets, labels, DEFAULT_LAMBDA);
    }

    /**
     * @return empty if there is no data or the optimisation does not produce a usable temperature
     */
    public static Optional<Calibrator.Bias> fit(List<double[]> rawLogProbs, int[] targets, List<String> labels,
                                                double lambda) {
        if (rawLogProbs.size() != targets.length) throw new IllegalArgumentException("size mismatch");
        int k = labels.size();
        int n = targets.length;
        if (n == 0 || k < 2) return Optional.empty();
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            if (rawLogProbs.get(i).length != k) throw new IllegalArgumentException("expected " + k + " scores");
            x[i] = finite(rawLogProbs.get(i));
        }

        // theta = (s, b_0 .. b_k-1); start from the temperature fit, which is the optimum for b = 0
        double[] theta = new double[k + 1];
        theta[0] = 1 / TemperatureFit.fit(rawLogProbs, targets);
        double loss = loss(theta, x, targets, lambda);
        for (int iter = 0; iter < 100; iter++) {
            double[] g = new double[k + 1];
            double[][] h = new double[k + 1][k + 1];
            derivatives(theta, x, targets, lambda, g, h);
            double gmax = 0;
            for (double v : g) gmax = Math.max(gmax, Math.abs(v));
            if (gmax < 1e-9) break;

            double[] step = solve(h, g);
            if (step == null) break;
            double alpha = 1;
            boolean improved = false;
            for (int ls = 0; ls < 30; ls++, alpha /= 2) {
                double[] next = theta.clone();
                for (int j = 0; j <= k; j++) next[j] -= alpha * step[j];
                next[0] = Math.max(1 / TemperatureFit.MAX_T, Math.min(1 / TemperatureFit.MIN_T, next[0]));
                double l = loss(next, x, targets, lambda);
                if (l < loss) {
                    theta = next;
                    improved = loss - l > 1e-13;
                    loss = l;
                    break;
                }
            }
            if (!improved) break;
        }
        if (!(theta[0] > 0) || !Double.isFinite(theta[0])) return Optional.empty();
        Map<String, Double> bias = new LinkedHashMap<>();
        for (int j = 0; j < k; j++) {
            if (!Double.isFinite(theta[j + 1])) return Optional.empty();
            bias.put(labels.get(j), theta[j + 1]);
        }
        return Optional.of(new Calibrator.Bias(1 / theta[0], bias));
    }

    /** -Infinity (a label the backend never saw) becomes a large finite gap, so derivatives stay defined. */
    private static double[] finite(double[] raw) {
        double min = Double.POSITIVE_INFINITY;
        for (double v : raw) if (Double.isFinite(v)) min = Math.min(min, v);
        double[] out = raw.clone();
        for (int j = 0; j < out.length; j++) {
            if (!Double.isFinite(out[j])) out[j] = min == Double.POSITIVE_INFINITY ? 0 : min - 30;
        }
        return out;
    }

    private static double[] probs(double[] theta, double[] xi) {
        int k = xi.length;
        double[] z = new double[k];
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < k; j++) {
            z[j] = theta[0] * xi[j] + theta[j + 1];
            max = Math.max(max, z[j]);
        }
        double sum = 0;
        for (int j = 0; j < k; j++) {
            z[j] = Math.exp(z[j] - max);
            sum += z[j];
        }
        for (int j = 0; j < k; j++) z[j] /= sum;
        return z;
    }

    static double loss(double[] theta, double[][] x, int[] targets, double lambda) {
        double sum = 0;
        for (int i = 0; i < targets.length; i++) sum -= Math.log(Math.max(probs(theta, x[i])[targets[i]], 1e-300));
        double reg = 0;
        for (int j = 1; j < theta.length; j++) reg += theta[j] * theta[j];
        return sum / targets.length + lambda / 2 * reg;
    }

    private static void derivatives(double[] theta, double[][] x, int[] targets, double lambda, double[] g, double[][] h) {
        int k = theta.length - 1;
        int n = targets.length;
        for (int i = 0; i < n; i++) {
            double[] p = probs(theta, x[i]);
            double[] xi = x[i];
            double px = 0, pxx = 0;
            for (int j = 0; j < k; j++) {
                px += p[j] * xi[j];
                pxx += p[j] * xi[j] * xi[j];
            }
            g[0] += px - xi[targets[i]];
            h[0][0] += pxx - px * px;
            for (int j = 0; j < k; j++) {
                g[j + 1] += p[j] - (j == targets[i] ? 1 : 0);
                double hs = p[j] * xi[j] - p[j] * px;
                h[0][j + 1] += hs;
                h[j + 1][0] += hs;
                for (int m = 0; m < k; m++) h[j + 1][m + 1] += (j == m ? p[j] : 0) - p[j] * p[m];
            }
        }
        for (int a = 0; a <= k; a++) {
            g[a] /= n;
            for (int c = 0; c <= k; c++) h[a][c] /= n;
        }
        for (int j = 1; j <= k; j++) {
            g[j] += lambda * theta[j];
            h[j][j] += lambda;
        }
        h[0][0] += 1e-9;
    }

    /** Gaussian elimination with partial pivoting; null if singular. */
    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
            if (Math.abs(m[pivot][col]) < 1e-15) return null;
            double[] tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp;
            for (int r = col + 1; r < n; r++) {
                double f = m[r][col] / m[col][col];
                for (int c = col; c <= n; c++) m[r][c] -= f * m[col][c];
            }
        }
        double[] out = new double[n];
        for (int r = n - 1; r >= 0; r--) {
            double v = m[r][n];
            for (int c = r + 1; c < n; c++) v -= m[r][c] * out[c];
            out[r] = v / m[r][r];
        }
        return out;
    }
}
