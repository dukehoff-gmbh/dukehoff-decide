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

/** Per-field result with the full distribution, not just the winner. */
public sealed interface FieldResult {

    String name();

    /** Probability of the returned value (calibrated if a profile was available). */
    double confidence();

    /** Probabilities per option, in {@link com.dukehoff.decide.spi.FieldSpec#options()} order. */
    double[] probabilities();

    record Choice<E extends Enum<E>>(String name, E value, Map<E, Double> distribution, double[] probabilities)
            implements FieldResult {
        public double confidence() { return distribution.get(value); }
    }

    record Bool(String name, boolean value, double probabilityTrue, double[] probabilities) implements FieldResult {
        public double confidence() { return value ? probabilityTrue : 1 - probabilityTrue; }
    }

    /**
     * @param value  probability-weighted mean over the scale anchors
     * @param spread standard deviation divided by the maximum possible one for the scale (0..1);
     *               high spread means "the model is torn"
     */
    record Score(String name, double value, double spread, double[] probabilities) implements FieldResult {
        public double confidence() { return 1 - spread; }
    }
}
