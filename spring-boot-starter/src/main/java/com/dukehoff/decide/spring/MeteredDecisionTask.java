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
package com.dukehoff.decide.spring;

import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Outcome;
import com.dukehoff.decide.State;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.TaskSchema;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/**
 * Records, tagged with {@code task} (the output type's name) and {@code model}:
 * <ul>
 *   <li>{@code dukehoff.decide.decisions}: timer per call, tagged {@code mode} (single, batch) and
 *       {@code outcome} (success, error); a batch is one sample</li>
 *   <li>{@code dukehoff.decide.decision.confidence}: distribution of {@link Decision#confidence()}</li>
 *   <li>{@code dukehoff.decide.gated}: counter of gated decisions, tagged {@code outcome} (confident, uncertain);
 *       uncertain / total is the deferral rate</li>
 * </ul>
 * {@code rawScores} is not recorded; it is for calibration tooling.
 */
final class MeteredDecisionTask<T> implements DecisionTask<T> {

    private final DecisionTask<T> delegate;
    private final MeterRegistry registry;
    private final Tags tags;
    private final DistributionSummary confidence;

    MeteredDecisionTask(DecisionTask<T> delegate, MeterRegistry registry) {
        this.delegate = delegate;
        this.registry = registry;
        this.tags = Tags.of("task", delegate.id(), "model", delegate.modelId());
        this.confidence = DistributionSummary.builder("dukehoff.decide.decision.confidence")
                .description("Confidence of each decision (the minimum over its fields)")
                .tags(tags)
                .register(registry);
    }

    @Override
    public String id() {
        return delegate.id();
    }

    @Override
    public Decision<T> decide(State state) {
        return timed("single", () -> recorded(delegate.decide(state)));
    }

    @Override
    public CompletableFuture<Decision<T>> decideAsync(State state) {
        Timer.Sample sample = Timer.start(registry);
        return delegate.decideAsync(state).whenComplete((d, e) -> {
            sample.stop(timer("single", e == null));
            if (d != null) recorded(d);
        });
    }

    @Override
    public List<Decision<T>> decideAll(List<State> states) {
        return timed("batch", () -> {
            List<Decision<T>> decisions = delegate.decideAll(states);
            decisions.forEach(this::recorded);
            return decisions;
        });
    }

    @Override
    public Gated<T> requireConfidence(double threshold) {
        return gated(delegate.requireConfidence(threshold));
    }

    @Override
    public Gated<T> requireConfidence(Map<String, Double> thresholdsByField) {
        return gated(delegate.requireConfidence(thresholdsByField));
    }

    @Override
    public TaskSchema schema() {
        return delegate.schema();
    }

    @Override
    public String modelId() {
        return delegate.modelId();
    }

    @Override
    public Map<String, RawScores> rawScores(State state) {
        return delegate.rawScores(state);
    }

    @Override
    public List<Map<String, RawScores>> rawScores(List<State> states) {
        return delegate.rawScores(states);
    }

    private Gated<T> gated(Gated<T> gated) {
        Counter confident = gatedCounter("confident");
        Counter uncertain = gatedCounter("uncertain");
        return state -> timed("single", () -> {
            Outcome<T> outcome = gated.decide(state);
            recorded(outcome.decision());
            (outcome instanceof Outcome.Confident<T> ? confident : uncertain).increment();
            return outcome;
        });
    }

    private Counter gatedCounter(String outcome) {
        return Counter.builder("dukehoff.decide.gated")
                .description("Gated decisions; uncertain ones are deferred to the caller")
                .tags(tags).tag("outcome", outcome)
                .register(registry);
    }

    private Decision<T> recorded(Decision<T> d) {
        confidence.record(d.confidence());
        return d;
    }

    private <R> R timed(String mode, Supplier<R> call) {
        Timer.Sample sample = Timer.start(registry);
        boolean ok = false;
        try {
            R result = call.get();
            ok = true;
            return result;
        } finally {
            sample.stop(timer(mode, ok));
        }
    }

    private Timer timer(String mode, boolean success) {
        return Timer.builder("dukehoff.decide.decisions")
                .description("Time to decide, including all model calls")
                .tags(tags).tag("mode", mode).tag("outcome", success ? "success" : "error")
                .register(registry);
    }
}
