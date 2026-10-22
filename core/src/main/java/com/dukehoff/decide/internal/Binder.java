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
import com.dukehoff.decide.FieldResult;
import com.dukehoff.decide.calibration.Metrics;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.TaskSchema;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Converts per-field probabilities into typed values, and typed values back into option indices. */
public final class Binder {

    private Binder() {}

    public record Bound<T>(T value, Map<String, FieldResult> fields) {}

    /** @param probs calibrated probabilities per field, in option order */
    @SuppressWarnings("unchecked")
    public static <T> Bound<T> bind(TaskSchema schema, Map<String, double[]> probs) {
        Map<String, FieldResult> results = new LinkedHashMap<>();
        for (FieldSpec f : schema.fields()) {
            results.put(f.name(), result(f, probs.get(f.name())));
        }
        T value = switch (schema.kind()) {
            case BOOLEAN -> (T) (Boolean) ((FieldResult.Bool) results.values().iterator().next()).value();
            case ENUM -> (T) ((FieldResult.Choice<?>) results.values().iterator().next()).value();
            case RECORD -> (T) construct(schema, results);
        };
        return new Bound<>(value, results);
    }

    /** The result of one field for calibrated probabilities {@code p}; also used to evaluate gating offline. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static FieldResult result(FieldSpec spec, double[] p) {
        return switch (spec) {
            case FieldSpec.BooleanSpec b -> new FieldResult.Bool(b.name(), p[0] >= p[1], p[0], p);
            case FieldSpec.ChoiceSpec c -> {
                Object[] constants = c.enumType().getEnumConstants();
                Map<Enum, Double> dist = new LinkedHashMap<>();
                for (int i = 0; i < constants.length; i++) dist.put((Enum) constants[i], p[i]);
                Enum winner = (Enum) constants[Metrics.argmax(p)];
                yield new FieldResult.Choice(c.name(), winner, Collections.unmodifiableMap(dist), p);
            }
            case FieldSpec.ScoreSpec s -> {
                List<Double> v = s.anchorValues();
                double mean = 0;
                for (int i = 0; i < p.length; i++) mean += p[i] * v.get(i);
                double var = 0;
                for (int i = 0; i < p.length; i++) var += p[i] * (v.get(i) - mean) * (v.get(i) - mean);
                double maxSd = (s.max() - s.min()) / 2;
                double spread = maxSd == 0 ? 0 : Math.min(1, Math.sqrt(var) / maxSd);
                yield new FieldResult.Score(s.name(), mean, spread, p);
            }
        };
    }

    private static Object construct(TaskSchema schema, Map<String, FieldResult> results) {
        RecordComponent[] components = schema.type().getRecordComponents();
        Class<?>[] types = new Class<?>[components.length];
        Object[] args = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            types[i] = components[i].getType();
            args[i] = toJava(results.get(components[i].getName()), types[i]);
        }
        try {
            Constructor<?> ctor = schema.type().getDeclaredConstructor(types);
            ctor.setAccessible(true);
            return ctor.newInstance(args);
        } catch (InvocationTargetException e) {
            throw new DecideException("Constructor of " + schema.type().getSimpleName()
                    + " rejected the decided values", e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new DecideException("Cannot construct " + schema.type().getName(), e);
        }
    }

    private static Object toJava(FieldResult r, Class<?> t) {
        return switch (r) {
            case FieldResult.Bool b -> b.value();
            case FieldResult.Choice<?> c -> c.value();
            case FieldResult.Score s -> {
                double v = s.value();
                if (t == double.class || t == Double.class) yield v;
                if (t == float.class || t == Float.class) yield (float) v;
                if (t == int.class || t == Integer.class) yield (int) Math.round(v);
                if (t == long.class || t == Long.class) yield Math.round(v);
                throw new IllegalStateException("unexpected type " + t);
            }
        };
    }

    /** The option index each field of {@code expected} corresponds to. Used for calibration. */
    public static Map<String, Integer> targetIndices(TaskSchema schema, Object expected) {
        if (expected == null) throw new IllegalArgumentException("expected value must not be null");
        Map<String, Integer> out = new LinkedHashMap<>();
        switch (schema.kind()) {
            case BOOLEAN, ENUM -> out.put(schema.fields().getFirst().name(),
                    index(schema.fields().getFirst(), expected));
            case RECORD -> {
                for (RecordComponent c : schema.type().getRecordComponents()) {
                    Object v;
                    try {
                        var accessor = c.getAccessor();
                        accessor.setAccessible(true);
                        v = accessor.invoke(expected);
                    } catch (ReflectiveOperationException e) {
                        throw new DecideException("Cannot read " + c.getName(), e);
                    }
                    out.put(c.getName(), index(schema.field(c.getName()), v));
                }
            }
        }
        return out;
    }

    private static int index(FieldSpec spec, Object v) {
        return switch (spec) {
            case FieldSpec.BooleanSpec b -> ((Boolean) v) ? 0 : 1;
            case FieldSpec.ChoiceSpec c -> ((Enum<?>) v).ordinal();
            case FieldSpec.ScoreSpec s -> {
                double x = ((Number) v).doubleValue();
                int best = 0;
                for (int i = 1; i < s.anchorValues().size(); i++) {
                    if (Math.abs(s.anchorValues().get(i) - x) < Math.abs(s.anchorValues().get(best) - x)) best = i;
                }
                yield best;
            }
        };
    }
}
