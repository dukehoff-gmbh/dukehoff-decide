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
package com.dukehoff.decide.internal;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.FieldResult;
import com.dukehoff.decide.Outcome;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibrator;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.TaskSchema;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class DefaultDecisionTask<T> implements DecisionTask<T> {

    private static final ExecutorService ASYNC = Executors.newVirtualThreadPerTaskExecutor();

    private final TaskSchema schema;
    private final ScoringBackend backend;
    private final Map<String, Calibrator> calibrators;
    private final boolean calibrated;

    DefaultDecisionTask(TaskSchema schema, ScoringBackend backend, Map<String, Calibrator> calibrators,
                        boolean calibrated) {
        this.schema = schema;
        this.backend = backend;
        this.calibrators = Map.copyOf(calibrators);
        this.calibrated = calibrated;
    }

    @Override
    public String id() { return schema.taskId(); }

    @Override
    public TaskSchema schema() { return schema; }

    @Override
    public String modelId() { return backend.modelId(); }

    @Override
    public Decision<T> decide(State state) {
        return toDecision(rawScores(state));
    }

    @Override
    public CompletableFuture<Decision<T>> decideAsync(State state) {
        return CompletableFuture.supplyAsync(() -> decide(state), ASYNC);
    }

    @Override
    public List<Decision<T>> decideAll(List<State> states) {
        return rawScores(states).stream().map(this::toDecision).toList();
    }

    @Override
    public Map<String, RawScores> rawScores(State state) {
        Map<String, RawScores> raw;
        try {
            raw = backend.score(new ScoringRequest(state, schema));
        } catch (DecideException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new DecideException("Backend " + backend.modelId() + " failed for " + schema.taskId(), e);
        }
        validate(raw);
        return raw;
    }

    @Override
    public List<Map<String, RawScores>> rawScores(List<State> states) {
        List<ScoringRequest> requests = states.stream().map(s -> new ScoringRequest(s, schema)).toList();
        List<Map<String, RawScores>> all;
        try {
            all = backend.scoreAll(requests);
        } catch (DecideException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new DecideException("Backend " + backend.modelId() + " failed for " + schema.taskId(), e);
        }
        if (all.size() != states.size()) {
            throw new DecideException("Backend returned " + all.size() + " results for " + states.size() + " states");
        }
        all.forEach(this::validate);
        return all;
    }

    private void validate(Map<String, RawScores> raw) {
        if (raw == null) throw new DecideException("Backend " + backend.modelId() + " returned null");
        for (FieldSpec f : schema.fields()) {
            RawScores r = raw.get(f.name());
            if (r == null) {
                throw new DecideException("Backend " + backend.modelId() + " returned no scores for field " + f.name());
            }
            if (r.logProbs().length != f.options().size()) {
                throw new DecideException("Backend " + backend.modelId() + " returned " + r.logProbs().length
                        + " scores for field " + f.name() + ", expected " + f.options().size());
            }
            for (double v : r.logProbs()) {
                if (Double.isNaN(v) || v == Double.POSITIVE_INFINITY) {
                    throw new DecideException("Backend returned invalid score " + v + " for field " + f.name());
                }
            }
        }
    }

    private Decision<T> toDecision(Map<String, RawScores> raw) {
        Map<String, double[]> probs = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            probs.put(f.name(), calibrators.get(f.name()).apply(raw.get(f.name()).logProbs()));
        }
        Binder.Bound<T> bound = Binder.bind(schema, probs);
        return new Decision<>(bound.value(), bound.fields(), calibrated);
    }

    @Override
    public Gated<T> requireConfidence(double threshold) {
        Map<String, Double> all = new LinkedHashMap<>();
        schema.fields().forEach(f -> all.put(f.name(), threshold));
        return requireConfidence(all);
    }

    @Override
    public Gated<T> requireConfidence(Map<String, Double> thresholdsByField) {
        Map<String, Double> thresholds = new LinkedHashMap<>(thresholdsByField);
        for (var e : thresholds.entrySet()) {
            schema.field(e.getKey()); // throws for unknown fields
            double t = e.getValue();
            if (t < 0 || t > 1) throw new IllegalArgumentException("threshold must be within 0..1: " + e);
        }
        return state -> {
            Decision<T> d = decide(state);
            List<String> uncertain = new ArrayList<>();
            for (FieldSpec f : schema.fields()) {
                Double t = thresholds.get(f.name());
                FieldResult r = d.fields().get(f.name());
                if (t != null && r.confidence() < t) uncertain.add(f.name());
            }
            return uncertain.isEmpty() ? new Outcome.Confident<>(d) : new Outcome.Uncertain<>(d, uncertain);
        };
    }
}
