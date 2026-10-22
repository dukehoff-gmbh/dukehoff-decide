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
package com.dukehoff.decide.testing;

import com.dukehoff.decide.State;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A backend driven by a function, for unit tests of code that uses dukehoff.decide.
 *
 * <pre>
 * var backend = new ScriptedBackend("test", (state, field) -&gt;
 *     state.text().contains("down") &amp;&amp; field.name().equals("priority")
 *         ? ScriptedBackend.probabilities(0.9, 0.05, 0.03, 0.02)
 *         : ScriptedBackend.uniform(field));
 * </pre>
 */
public final class ScriptedBackend implements ScoringBackend {

    @FunctionalInterface
    public interface Script {
        /** @return log-probabilities over {@code field.options()} */
        double[] score(State state, FieldSpec field);
    }

    private final String modelId;
    private final Script script;
    private final AtomicInteger calls = new AtomicInteger();

    public ScriptedBackend(String modelId, Script script) {
        this.modelId = modelId;
        this.script = script;
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        calls.incrementAndGet();
        Map<String, RawScores> out = new LinkedHashMap<>();
        for (FieldSpec f : request.schema().fields()) {
            out.put(f.name(), new RawScores(script.score(request.state(), f)));
        }
        return out;
    }

    /** Number of {@link #score} calls so far. */
    public int calls() {
        return calls.get();
    }

    /** Converts probabilities to log-probabilities. */
    public static double[] probabilities(double... p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) out[i] = Math.log(p[i]);
        return out;
    }

    public static double[] uniform(FieldSpec field) {
        return new double[field.options().size()];
    }
}
