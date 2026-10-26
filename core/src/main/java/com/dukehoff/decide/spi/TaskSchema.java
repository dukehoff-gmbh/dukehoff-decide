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

import com.dukehoff.decide.internal.Binder;
import com.dukehoff.decide.internal.SchemaReader;

import java.util.List;
import java.util.Map;

/**
 * Everything a backend needs to know about a task, independent of any input.
 *
 * @param taskId       stable id, the output type's fully qualified name
 * @param type         the output type
 * @param kind         how the output type maps to fields
 * @param instructions shared context for all questions; may be empty
 * @param fields       the questions, in record component order
 */
public record TaskSchema(String taskId, Class<?> type, Kind kind, String instructions, List<FieldSpec> fields) {

    public enum Kind { RECORD, ENUM, BOOLEAN }

    public TaskSchema {
        fields = List.copyOf(fields);
    }

    /**
     * The schema of an output type, as {@code decider.task(type)} reads it. For backends that need it before
     * any decision, e.g. to train a classifier.
     */
    public static TaskSchema of(Class<?> type) {
        return of(type, null);
    }

    /** As {@code decider.task(type, instructions)} reads it; a {@code Boolean} task needs the question here. */
    public static TaskSchema of(Class<?> type, String instructions) {
        return SchemaReader.read(type, instructions);
    }

    /**
     * The option index (in {@link FieldSpec#options()} order) that each field of {@code expected} corresponds
     * to, keyed by field name. A {@code @Scale} value maps to its nearest step.
     */
    public Map<String, Integer> targetIndices(Object expected) {
        return Binder.targetIndices(this, expected);
    }

    public FieldSpec field(String name) {
        return fields.stream().filter(f -> f.name().equals(name)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("No field '" + name + "' in " + taskId));
    }
}
