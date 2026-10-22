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

import java.util.Map;

/**
 * A typed result plus per-field evidence.
 *
 * @param value      the fully constructed record/enum/boolean - never null, never malformed
 * @param fields     per-field results keyed by record component name ("value" for enum/boolean tasks)
 * @param calibrated false if no calibration profile was found and raw model probabilities were used
 */
public record Decision<T>(T value, Map<String, FieldResult> fields, boolean calibrated) {

    /** Name of the single field of an enum or boolean task. */
    public static final String VALUE = "value";

    public Decision {
        fields = Map.copyOf(fields);
    }

    /** Overall confidence: the weakest field. Conservative on purpose. */
    public double confidence() {
        return fields.values().stream().mapToDouble(FieldResult::confidence).min().orElse(0);
    }

    public FieldResult field(String name) {
        FieldResult r = fields.get(name);
        if (r == null) throw new IllegalArgumentException("No field '" + name + "', have " + fields.keySet());
        return r;
    }

    /** Distribution of an enum field. */
    @SuppressWarnings("unchecked")
    public <E extends Enum<E>> Map<E, Double> distribution(String field) {
        if (field(field) instanceof FieldResult.Choice<?> c) {
            return (Map<E, Double>) c.distribution();
        }
        throw new IllegalArgumentException("Field '" + field + "' is not an enum field");
    }

    /** Distribution of an enum task, e.g. {@code decider.task(Priority.class, "...")}. */
    public <E extends Enum<E>> Map<E, Double> distribution() {
        return distribution(VALUE);
    }
}
