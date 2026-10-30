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
package com.dukehoff.decide.spi;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A backend that asks several backends and averages their normalised log-probabilities per option (a geometric
 * mean of their probabilities). Works best with backends that err on different cases, such as a
 * {@code LinearClassifier} (word counts) and an {@code EmbeddingClassifier} (meaning).
 *
 * <pre>{@code
 * ScoringBackend both = Ensemble.of(linear, embedding);
 * DecisionTask<Triage> raw = Decider.builder().backend(both).build().task(Triage.class);
 * profiles.save(Calibration.fit(raw, train, holdout).profile());   // calibrate the ensemble as a whole
 * }</pre>
 *
 * The ensemble is calibrated like any other backend, under its own {@link #modelId()}; the parts' calibration
 * profiles are not used. Every request goes to every part, through their {@link #scoreAll}, so batching parts
 * still batch. For a cheaper alternative that only asks the next backend when the previous one is unsure, see
 * {@link com.dukehoff.decide.Cascade}.
 */
public final class Ensemble implements ScoringBackend {

    private final List<ScoringBackend> parts;
    private final String modelId;

    private Ensemble(List<ScoringBackend> parts) {
        if (parts.size() < 2) throw new IllegalArgumentException("an ensemble needs at least two backends");
        this.parts = List.copyOf(parts);
        this.modelId = "ensemble(" + this.parts.stream().map(ScoringBackend::modelId).collect(Collectors.joining(",")) + ")";
    }

    public static Ensemble of(ScoringBackend... parts) {
        return new Ensemble(List.of(parts));
    }

    public static Ensemble of(List<? extends ScoringBackend> parts) {
        return new Ensemble(List.copyOf(parts));
    }

    /** {@code ensemble(<id>,<id>,...)}: changes when any part changes, which invalidates the calibration. */
    @Override
    public String modelId() {
        return modelId;
    }

    public List<ScoringBackend> parts() {
        return parts;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        return scoreAll(List.of(request)).getFirst();
    }

    @Override
    public List<Map<String, RawScores>> scoreAll(List<ScoringRequest> requests) {
        List<List<Map<String, RawScores>>> byPart = parts.stream().map(p -> p.scoreAll(requests)).toList();
        List<Map<String, RawScores>> out = new ArrayList<>(requests.size());
        for (int i = 0; i < requests.size(); i++) {
            Map<String, RawScores> fields = new LinkedHashMap<>();
            for (FieldSpec f : requests.get(i).schema().fields()) {
                double[] mean = new double[f.options().size()];
                for (List<Map<String, RawScores>> part : byPart) {
                    double[] lp = logSoftmax(part.get(i).get(f.name()).logProbs());
                    if (lp.length != mean.length) {
                        throw new IllegalStateException("backend returned " + lp.length + " scores for field "
                                + f.name() + ", which has " + mean.length + " options");
                    }
                    for (int k = 0; k < mean.length; k++) mean[k] += lp[k] / parts.size();
                }
                fields.put(f.name(), new RawScores(mean));
            }
            out.add(fields);
        }
        return out;
    }

    private static double[] logSoftmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) max = Math.max(max, v);
        double sum = 0;
        for (double v : z) sum += Math.exp(v - max);
        double log = max + Math.log(sum);
        double[] out = new double[z.length];
        for (int k = 0; k < z.length; k++) out[k] = z[k] - log;
        return out;
    }

    @Override
    public String toString() {
        return modelId;
    }
}
