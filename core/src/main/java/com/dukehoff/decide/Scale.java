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

import java.lang.annotation.*;

/**
 * Turns an int/double component into a score question.
 * Internally the range is discretised into {@code steps} anchors; the result is the
 * probability-weighted mean, and the spread is reported as uncertainty.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.RECORD_COMPONENT)
public @interface Scale {
    double min() default 0;
    double max() default 1;
    int steps() default 5;
    /** Optional meaning of the ends, e.g. {"not relevant", "exactly what was asked"}. */
    String[] anchors() default {};
}
