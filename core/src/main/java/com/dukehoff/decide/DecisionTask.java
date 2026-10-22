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

import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.TaskSchema;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * A prepared, reusable decision for one output type. Created once (schema is
 * reflected and validated, calibration is loaded), then called often. Thread-safe.
 */
public interface DecisionTask<T> {

    /** Stable id used to look up calibration profiles; the type's fully qualified name. */
    String id();

    Decision<T> decide(State state);

    CompletableFuture<Decision<T>> decideAsync(State state);

    /** Backends may batch; order of results matches order of input. */
    List<Decision<T>> decideAll(List<State> states);

    /** Same threshold for all fields. */
    Gated<T> requireConfidence(double threshold);

    /**
     * Per-field thresholds, e.g. churnRisk must be 0.95 but priority 0.7 is fine.
     * Fields not in the map are not gated.
     */
    Gated<T> requireConfidence(Map<String, Double> thresholdsByField);

    /** The questions this task asks. For tooling, calibration and debugging. */
    TaskSchema schema();

    /** Id of the model behind this task; calibration profiles are only valid for it. */
    String modelId();

    /** Uncalibrated backend output. For calibration and debugging. */
    Map<String, RawScores> rawScores(State state);

    /** Uncalibrated backend output for many states. */
    List<Map<String, RawScores>> rawScores(List<State> states);

    interface Gated<T> {
        Outcome<T> decide(State state);
    }
}
