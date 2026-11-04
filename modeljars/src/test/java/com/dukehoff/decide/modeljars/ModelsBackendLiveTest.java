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
package com.dukehoff.decide.modeljars;

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.State;
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.runtime.InferencePipeline;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Runs only with {@code DUKEHOFF_DECIDE_GGUF} pointing at a Qwen2.5 Instruct GGUF file. */
@EnabledIfEnvironmentVariable(named = "DUKEHOFF_DECIDE_GGUF", matches = ".+")
class ModelsBackendLiveTest {

    @Question("How urgent is this ticket?")
    enum Urgency {
        @Option("production is down or data is lost") BLOCKER,
        @Option("a feature is broken, no workaround") HIGH,
        @Option("a question or a minor issue") LOW
    }

    @Test
    void scoresAllLabelsFromTheFullVocabulary() {
        Path gguf = Path.of(System.getenv("DUKEHOFF_DECIDE_GGUF"));
        try (ModelsBackend backend = builder(gguf).build()) {
            var task = Decider.builder().backend(backend).build().task(Urgency.class);
            long start = System.nanoTime();
            Map<String, RawScores> raw = task.rawScores(State.of("Our production database is down since 9:00, nobody can log in."));
            System.out.printf("%s: %s in %d ms%n", backend.modelId(),
                    java.util.Arrays.toString(raw.values().iterator().next().logProbs()), (System.nanoTime() - start) / 1_000_000);
            double[] lp = raw.values().iterator().next().logProbs();
            assertEquals(3, lp.length);
            for (double v : lp) assertTrue(Double.isFinite(v), "every label has a score");
            for (String text : java.util.List.of("The invoice PDF has a typo in our street name.",
                    "Checkout fails for every customer with a 500 error.", "How do I change my password?")) {
                long t0 = System.nanoTime();
                task.rawScores(State.of(text));
                System.out.printf("warm: %d ms%n", (System.nanoTime() - t0) / 1_000_000);
            }
            assertEquals(Urgency.BLOCKER, task.decide(State.of("Our production database is down since 9:00, nobody can log in.")).value());
        }
    }

    @Test
    void reusingThePrefixGivesTheSameScores() {
        Path gguf = Path.of(System.getenv("DUKEHOFF_DECIDE_GGUF"));
        List<String> texts = List.of("Checkout fails for every customer with a 500 error.", "How do I change my password?",
                "How do I change my e-mail address?");
        try (ModelsBackend reuse = builder(gguf).build(); ModelsBackend fresh = builder(gguf).reusePrefix(false).build()) {
            var a = Decider.builder().backend(reuse).build().task(Urgency.class);
            var b = Decider.builder().backend(fresh).build().task(Urgency.class);
            for (String text : texts) {
                double[] x = a.rawScores(State.of(text)).get(Decision.VALUE).logProbs();
                double[] y = b.rawScores(State.of(text)).get(Decision.VALUE).logProbs();
                for (int k = 0; k < x.length; k++) assertEquals(y[k], x[k], 1e-3, text);
            }
        }
    }

    private static ModelsBackend.Builder builder(Path gguf) {
        return System.getenv("DUKEHOFF_DECIDE_NATIVE") != null
                ? ModelsBackend.builder().pipeline(new InferencePipeline(RustFfmBackend.load(gguf)))
                        .modelId(gguf.getFileName().toString().replace(".gguf", "") + "@native")
                : ModelsBackend.builder().model(gguf);
    }
}
