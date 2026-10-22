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

import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.internal.DefaultDecider;
import com.dukehoff.decide.spi.ScoringBackend;

/**
 * Entry point. Holds the backend and calibration store; hands out tasks.
 *
 * <pre>
 * Decider decider = Decider.builder()
 *     .backend(myBackend)
 *     .calibration(CalibrationStore.directory(Path.of("calibration")))
 *     .build();
 *
 * DecisionTask&lt;Triage&gt; triage = decider.task(Triage.class);
 * </pre>
 *
 * Supported output types:
 * <ul>
 *   <li>a record whose components are enums, booleans, or numbers annotated with {@link Scale}</li>
 *   <li>an enum (single question)</li>
 *   <li>{@code Boolean.class} (single yes/no question; instructions required)</li>
 * </ul>
 */
public interface Decider {

    <T> DecisionTask<T> task(Class<T> type);

    /** Ad-hoc instructions: the question for enum/boolean tasks, extra context for records. */
    <T> DecisionTask<T> task(Class<T> type, String instructions);

    static Builder builder() {
        return new DefaultDecider.Builder();
    }

    interface Builder {
        Builder backend(ScoringBackend backend);

        Builder calibration(CalibrationStore store);

        /** If true, creating a task without a matching profile fails. Default false. */
        Builder requireCalibration(boolean required);

        Decider build();
    }
}
