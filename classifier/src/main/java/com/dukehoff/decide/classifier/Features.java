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
package com.dukehoff.decide.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Text to hashed TF-IDF features: lower-cased word 1-2-grams and character 3-5-grams (over the words joined by
 * single spaces), hashed into {@code dim} buckets, sublinear term frequency, L2-normalised.
 */
final class Features {

    record Sparse(int[] index, double[] value) {}

    private Features() {}

    /** Hashed n-gram counts, keyed by bucket in ascending order. */
    static Map<Integer, Double> counts(String text, int dim) {
        String t = text.toLowerCase(Locale.ROOT);
        Map<Integer, Double> c = new TreeMap<>();
        List<String> words = new ArrayList<>(Arrays.asList(t.split("[^\\p{L}\\p{N}{}]+")));
        words.removeIf(String::isEmpty);
        for (int i = 0; i < words.size(); i++) {
            add(c, "w:" + words.get(i), dim);
            if (i > 0) add(c, "b:" + words.get(i - 1) + ' ' + words.get(i), dim);
        }
        String padded = ' ' + String.join(" ", words) + ' ';
        for (int n = 3; n <= 5; n++) {
            for (int i = 0; i + n <= padded.length(); i++) add(c, "c:" + padded.substring(i, i + n), dim);
        }
        return c;
    }

    private static void add(Map<Integer, Double> c, String feature, int dim) {
        c.merge(Math.floorMod(feature.hashCode() * 0x9E3779B1, dim), 1.0, Double::sum);
    }

    /** Sublinear TF times IDF, L2-normalised. */
    static Sparse tfidf(Map<Integer, Double> counts, float[] idf) {
        int[] index = new int[counts.size()];
        double[] value = new double[counts.size()];
        int j = 0;
        double norm = 0;
        for (Map.Entry<Integer, Double> e : counts.entrySet()) {
            index[j] = e.getKey();
            value[j] = (1 + Math.log(e.getValue())) * idf[e.getKey()];
            norm += value[j] * value[j];
            j++;
        }
        norm = Math.sqrt(norm);
        for (j = 0; j < value.length; j++) value[j] /= norm == 0 ? 1 : norm;
        return new Sparse(index, value);
    }

    static double[] logSoftmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0), sum = 0;
        for (double v : z) sum += Math.exp(v - max);
        double log = max + Math.log(sum);
        return Arrays.stream(z).map(v -> v - log).toArray();
    }

    static double[] softmax(double[] z) {
        double[] p = logSoftmax(z);
        for (int k = 0; k < p.length; k++) p[k] = Math.exp(p[k]);
        return p;
    }
}
