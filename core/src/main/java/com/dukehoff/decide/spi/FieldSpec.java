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

import java.util.List;

/**
 * Backend-neutral description of one question, derived from the Java type via reflection.
 * Every field is reduced to "pick one of N labelled options": booleans are 2 options
 * (yes, no), enums one option per constant in ordinal order, scales one option per anchor.
 * That keeps the backend contract tiny.
 */
public sealed interface FieldSpec {

    String name();

    String question();

    /** Options shown to the model, in index order. Backends must return scores in this order. */
    List<OptionSpec> options();

    /**
     * @param label       short identifier, e.g. the enum constant name, "yes" or "3"
     * @param description optional explanation for the model; may be empty
     */
    record OptionSpec(String label, String description) {}

    record BooleanSpec(String name, String question) implements FieldSpec {
        private static final List<OptionSpec> YES_NO =
                List.of(new OptionSpec("yes", ""), new OptionSpec("no", ""));

        public List<OptionSpec> options() { return YES_NO; }
    }

    record ChoiceSpec(String name, String question, Class<? extends Enum<?>> enumType, List<OptionSpec> options)
            implements FieldSpec {
        public ChoiceSpec {
            options = List.copyOf(options);
        }
    }

    /**
     * @param anchorValues numeric value represented by each option, ascending
     */
    record ScoreSpec(String name, String question, double min, double max, List<Double> anchorValues,
                     List<OptionSpec> options) implements FieldSpec {
        public ScoreSpec {
            anchorValues = List.copyOf(anchorValues);
            options = List.copyOf(options);
        }
    }
}
