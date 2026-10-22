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
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.Scale;
import com.dukehoff.decide.spi.FieldSpec.OptionSpec;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.TaskSchema;

import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Reflects an output type into a {@link TaskSchema}. Fails fast with actionable messages. */
public final class SchemaReader {

    private SchemaReader() {}

    public static TaskSchema read(Class<?> type, String instructions) {
        String extra = instructions == null ? "" : instructions.strip();
        if (type == Boolean.class || type == boolean.class) {
            if (extra.isEmpty()) {
                throw new DecideException("A Boolean task needs the question as instructions, "
                        + "e.g. decider.task(Boolean.class, \"Is this spam?\")");
            }
            return new TaskSchema(Boolean.class.getName(), Boolean.class, TaskSchema.Kind.BOOLEAN, "",
                    List.of(new FieldSpec.BooleanSpec(Decision.VALUE, extra)));
        }
        if (type.isEnum()) {
            String question = !extra.isEmpty() ? extra : questionOf(type);
            if (question == null) {
                throw new DecideException("Enum task " + type.getName()
                        + " needs a question: annotate the enum with @Question or pass instructions");
            }
            return new TaskSchema(type.getName(), type, TaskSchema.Kind.ENUM, "",
                    List.of(choice(Decision.VALUE, question, type)));
        }
        if (type.isRecord()) {
            String typeQuestion = questionOf(type);
            String shared = join(typeQuestion, extra);
            List<FieldSpec> fields = new ArrayList<>();
            for (RecordComponent c : type.getRecordComponents()) {
                fields.add(field(type, c));
            }
            if (fields.isEmpty()) throw new DecideException("Record " + type.getName() + " has no components");
            return new TaskSchema(type.getName(), type, TaskSchema.Kind.RECORD, shared, fields);
        }
        throw new DecideException("Unsupported output type " + type.getName()
                + ": use a record, an enum or Boolean");
    }

    private static FieldSpec field(Class<?> owner, RecordComponent c) {
        Question q = c.getAnnotation(Question.class);
        String where = owner.getSimpleName() + "." + c.getName();
        if (q == null || q.value().isBlank()) {
            throw new DecideException(where + " needs @Question(\"...\") - the model has to know what to decide");
        }
        Class<?> t = c.getType();
        Scale scale = c.getAnnotation(Scale.class);
        if (t == boolean.class || t == Boolean.class) {
            requireNoScale(where, scale);
            return new FieldSpec.BooleanSpec(c.getName(), q.value());
        }
        if (t.isEnum()) {
            requireNoScale(where, scale);
            return choice(c.getName(), q.value(), t);
        }
        if (isNumber(t)) {
            if (scale == null) {
                throw new DecideException(where + " is numeric and needs @Scale(min=.., max=.., steps=..)");
            }
            return score(where, c.getName(), q.value(), scale);
        }
        throw new DecideException(where + " has unsupported type " + t.getSimpleName()
                + ": use boolean, an enum or a number with @Scale");
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static FieldSpec.ChoiceSpec choice(String name, String question, Class<?> enumType) {
        Object[] constants = enumType.getEnumConstants();
        if (constants.length < 2) {
            throw new DecideException("Enum " + enumType.getName() + " needs at least two constants");
        }
        List<OptionSpec> options = new ArrayList<>();
        for (Object o : constants) {
            Enum<?> e = (Enum<?>) o;
            String description = "";
            try {
                Option opt = enumType.getField(e.name()).getAnnotation(Option.class);
                if (opt != null) description = opt.value();
            } catch (NoSuchFieldException ex) {
                throw new IllegalStateException(ex);
            }
            options.add(new OptionSpec(e.name(), description));
        }
        return new FieldSpec.ChoiceSpec(name, question, (Class) enumType, options);
    }

    private static FieldSpec.ScoreSpec score(String where, String name, String question, Scale s) {
        if (s.steps() < 2) throw new DecideException(where + ": @Scale steps must be >= 2");
        if (!(s.max() > s.min())) throw new DecideException(where + ": @Scale max must be > min");
        String[] anchors = s.anchors();
        if (anchors.length != 0 && anchors.length != 2 && anchors.length != s.steps()) {
            throw new DecideException(where + ": @Scale anchors must be empty, 2 (low/high end) or one per step");
        }
        List<Double> values = new ArrayList<>();
        List<OptionSpec> options = new ArrayList<>();
        for (int i = 0; i < s.steps(); i++) {
            double v = s.min() + i * (s.max() - s.min()) / (s.steps() - 1);
            values.add(v);
            String text = "";
            if (anchors.length == s.steps()) text = anchors[i];
            else if (anchors.length == 2 && i == 0) text = anchors[0];
            else if (anchors.length == 2 && i == s.steps() - 1) text = anchors[1];
            String description = String.format(Locale.ROOT, "%s%s", format(v), text.isEmpty() ? "" : " = " + text);
            options.add(new OptionSpec(Integer.toString(i + 1), description));
        }
        return new FieldSpec.ScoreSpec(name, question, s.min(), s.max(), values, options);
    }

    private static String format(double v) {
        return v == Math.rint(v) ? Long.toString((long) v) : String.format(Locale.ROOT, "%.3g", v);
    }

    private static void requireNoScale(String where, Scale scale) {
        if (scale != null) throw new DecideException(where + ": @Scale only applies to numeric components");
    }

    static boolean isNumber(Class<?> t) {
        return t == double.class || t == Double.class || t == float.class || t == Float.class
                || t == int.class || t == Integer.class || t == long.class || t == Long.class;
    }

    private static String questionOf(Class<?> type) {
        Question q = type.getAnnotation(Question.class);
        return q == null || q.value().isBlank() ? null : q.value();
    }

    private static String join(String a, String b) {
        if (a == null || a.isBlank()) return b;
        if (b == null || b.isBlank()) return a;
        return a + "\n" + b;
    }
}
