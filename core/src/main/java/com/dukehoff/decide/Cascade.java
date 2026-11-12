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
package com.dukehoff.decide;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Asks gated tasks in order and takes the first confident answer: a cheap backend decides the easy cases, the
 * next one only sees what the previous one was unsure about. Each stage is its own calibrated task with its own
 * threshold.
 *
 * <pre>{@code
 * Cascade<Triage> triage = Cascade.of(
 *         linearTask.requireConfidence(0.9),      // microseconds, decides most cases
 *         embeddingTask.requireConfidence(0.9),   // one embedding request, for the rest
 *         llmTask.requireConfidence(0.95));       // optional last resort
 * switch (triage.decide(state)) {
 *     case Outcome.Confident<Triage>(var d)       -> route(d.value());
 *     case Outcome.Uncertain<Triage>(var d, var f) -> reviewQueue.add(state, d);
 * }
 * }</pre>
 *
 * If no stage is confident, the result is the last stage's {@link Outcome.Uncertain}. A cascade's accuracy at a
 * threshold is usually a little lower than a single task's at the same threshold, because the later stages only
 * get the hard cases; pick thresholds by measuring the whole cascade on labelled data.
 */
public final class Cascade<T> implements DecisionTask.Gated<T> {

    /**
     * An outcome and the stage that produced it.
     *
     * @param stage index into {@link #stages()}; for an uncertain outcome, the last stage
     */
    public record Step<T>(Outcome<T> outcome, int stage) {}

    private final List<DecisionTask.Gated<T>> stages;

    private Cascade(List<DecisionTask.Gated<T>> stages) {
        if (stages.isEmpty()) throw new IllegalArgumentException("a cascade needs at least one stage");
        this.stages = List.copyOf(stages);
    }

    @SafeVarargs
    public static <T> Cascade<T> of(DecisionTask.Gated<T>... stages) {
        return new Cascade<>(List.of(stages));
    }

    public static <T> Cascade<T> of(List<? extends DecisionTask.Gated<T>> stages) {
        return new Cascade<>(List.copyOf(stages));
    }

    public List<DecisionTask.Gated<T>> stages() {
        return stages;
    }

    @Override
    public Outcome<T> decide(State state) {
        return step(state).outcome();
    }

    /** Like {@link #decide}, but also says which stage decided, e.g. for metrics on how often the fallback runs. */
    public Step<T> step(State state) {
        Outcome<T> outcome = null;
        for (int i = 0; i < stages.size(); i++) {
            outcome = stages.get(i).decide(state);
            if (outcome instanceof Outcome.Confident<T>) return new Step<>(outcome, i);
        }
        return new Step<>(outcome, stages.size() - 1);
    }

    /**
     * Each stage gets the states the stages before it were unsure about, in one {@link DecisionTask.Gated#decideAll}
     * call, so backends that batch (embedding requests, for example) still batch.
     */
    @Override
    public List<Outcome<T>> decideAll(List<State> states) {
        return stepAll(states).stream().map(Step::outcome).toList();
    }

    /** Like {@link #decideAll}, with the stage that decided each state; order matches the input. */
    public List<Step<T>> stepAll(List<State> states) {
        List<Step<T>> out = new ArrayList<>(Collections.nCopies(states.size(), null));
        List<Integer> open = new ArrayList<>();
        for (int i = 0; i < states.size(); i++) open.add(i);
        for (int stage = 0; stage < stages.size() && !open.isEmpty(); stage++) {
            List<Outcome<T>> outcomes = stages.get(stage).decideAll(open.stream().map(states::get).toList());
            List<Integer> unsure = new ArrayList<>();
            for (int k = 0; k < open.size(); k++) {
                Outcome<T> outcome = outcomes.get(k);
                out.set(open.get(k), new Step<>(outcome, stage));
                if (!(outcome instanceof Outcome.Confident<T>)) unsure.add(open.get(k));
            }
            open = unsure;
        }
        return out;
    }
}
