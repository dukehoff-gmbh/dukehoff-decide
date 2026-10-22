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
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.calibration.Calibrator;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.TaskSchema;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public final class DefaultDecider implements Decider {

    private final ScoringBackend backend;
    private final CalibrationStore store;
    private final boolean requireCalibration;

    private DefaultDecider(Builder b) {
        this.backend = Objects.requireNonNull(b.backend, "backend is required");
        this.store = b.store != null ? b.store : CalibrationStore.inMemory();
        this.requireCalibration = b.requireCalibration;
    }

    @Override
    public <T> DecisionTask<T> task(Class<T> type) {
        return task(type, "");
    }

    @Override
    public <T> DecisionTask<T> task(Class<T> type, String instructions) {
        TaskSchema schema = SchemaReader.read(type, instructions);
        Optional<Calibration.Profile> profile = store.load(schema.taskId(), backend.modelId());
        Map<String, Calibrator> calibrators = new LinkedHashMap<>();
        boolean complete = profile.isPresent();
        String mismatch = null;
        for (FieldSpec f : schema.fields()) {
            Calibrator stored = profile.map(p -> p.byField().get(f.name())).orElse(null);
            if (stored == null) {
                complete = false;
                continue;
            }
            List<String> labels = f.options().stream().map(FieldSpec.OptionSpec::label).toList();
            Optional<Calibrator> bound = stored.forOptions(labels);
            if (bound.isEmpty()) {
                complete = false;
                mismatch = "field '" + f.name() + "' was calibrated for other options than " + labels;
                continue;
            }
            calibrators.put(f.name(), bound.get());
        }
        if (mismatch != null) {
            // a profile for other options would mis-assign biases, so none of it is used
            System.getLogger(DefaultDecider.class.getName()).log(System.Logger.Level.WARNING,
                    "Ignoring calibration profile for " + schema.taskId() + " on model " + backend.modelId()
                            + ": " + mismatch + ". Fit and save a new profile.");
            calibrators.clear();
        }
        if (requireCalibration && !complete) {
            throw new DecideException("No complete calibration profile for " + schema.taskId()
                    + " on model " + backend.modelId()
                    + (mismatch != null ? " (" + mismatch + ")" : "") + ". Run Calibration.fit(...) and save the profile.");
        }
        if (complete) {
            List<String> changed = schema.fields().stream()
                    .filter(f -> {
                        String stored = profile.get().promptHashByField().get(f.name());
                        return stored != null && !stored.equals(LabelPrompt.fingerprint(schema, f));
                    })
                    .map(FieldSpec::name).toList();
            if (!changed.isEmpty()) {
                // still structurally valid, but fitted for other prompts: usable, though probably off
                String message = "Calibration profile for " + schema.taskId() + " on model " + backend.modelId()
                        + " was fitted for a different prompt of " + changed + " (an @Question or @Option text, the"
                        + " option order or the prompt template changed). Fit and save a new profile.";
                if (requireCalibration) throw new DecideException(message);
                System.getLogger(DefaultDecider.class.getName()).log(System.Logger.Level.WARNING, message);
            }
        }
        for (FieldSpec f : schema.fields()) calibrators.putIfAbsent(f.name(), Calibrator.identity());
        return new DefaultDecisionTask<>(schema, backend, calibrators, complete);
    }

    public static final class Builder implements Decider.Builder {
        private ScoringBackend backend;
        private CalibrationStore store;
        private boolean requireCalibration;

        @Override
        public Builder backend(ScoringBackend backend) { this.backend = backend; return this; }

        @Override
        public Builder calibration(CalibrationStore store) { this.store = store; return this; }

        @Override
        public Builder requireCalibration(boolean required) { this.requireCalibration = required; return this; }

        @Override
        public Decider build() { return new DefaultDecider(this); }
    }
}
