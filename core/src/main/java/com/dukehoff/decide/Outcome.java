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

import java.util.List;

/**
 * Result of a gated task: forces the caller to handle the uncertain case,
 * e.g. route it to a human or a bigger model.
 *
 * <pre>
 * switch (gated.decide(state)) {
 *     case Outcome.Confident&lt;Triage&gt;(var d)        -&gt; route(d.value());
 *     case Outcome.Uncertain&lt;Triage&gt;(var d, var why) -&gt; reviewQueue.add(state, d);
 * }
 * </pre>
 */
public sealed interface Outcome<T> {

    Decision<T> decision();

    record Confident<T>(Decision<T> decision) implements Outcome<T> {}

    /** @param uncertainFields fields below their threshold */
    record Uncertain<T>(Decision<T> decision, List<String> uncertainFields) implements Outcome<T> {
        public Uncertain {
            uncertainFields = List.copyOf(uncertainFields);
        }
    }
}
