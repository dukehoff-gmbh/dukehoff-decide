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

import com.dukehoff.decide.Fixtures.Priority;
import com.dukehoff.decide.spi.Ensemble;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static com.dukehoff.decide.testing.ScriptedBackend.probabilities;
import static org.junit.jupiter.api.Assertions.*;

class CombiningTest {

    /** Sure about "down", unsure about everything else. */
    final ScriptedBackend words = new ScriptedBackend("words", (state, field) -> state.text().contains("down")
            ? probabilities(0.90, 0.06, 0.03, 0.01) : probabilities(0.4, 0.3, 0.2, 0.1));

    /** Sure about "broken", unsure about everything else; scores need not be normalised. */
    final ScriptedBackend meaning = new ScriptedBackend("meaning", (state, field) -> state.text().contains("broken")
            ? new double[] {0, 5, 0, 0} : new double[] {1, 1, 2, 1});

    @Test
    void ensembleAveragesNormalisedLogProbabilities() {
        Ensemble both = Ensemble.of(words, meaning);
        assertEquals("ensemble(words,meaning)", both.modelId());

        Map<String, RawScores> raw = Decider.builder().backend(both).build().task(Priority.class)
                .rawScores(State.of("server down"));
        double[] lp = raw.get(Decision.VALUE).logProbs();
        double z = Math.log(3 * Math.E + Math.exp(2));
        double[] meaningLp = {1 - z, 1 - z, 2 - z, 1 - z};
        double[] wordsP = {0.90, 0.06, 0.03, 0.01};
        for (int k = 0; k < 4; k++) assertEquals((Math.log(wordsP[k]) + meaningLp[k]) / 2, lp[k], 1e-12);
    }

    @Test
    void ensembleNeedsTwoBackends() {
        assertThrows(IllegalArgumentException.class, () -> Ensemble.of(words));
    }

    @Test
    void cascadeTakesTheFirstConfidentStage() {
        Decider d1 = Decider.builder().backend(words).build();
        Decider d2 = Decider.builder().backend(meaning).build();
        Cascade<Priority> cascade = Cascade.of(
                d1.task(Priority.class).requireConfidence(0.8),
                d2.task(Priority.class).requireConfidence(0.8));

        int before = meaning.calls();
        Cascade.Step<Priority> down = cascade.step(State.of("server down"));
        assertEquals(0, down.stage());
        assertEquals(Priority.BLOCKER, assertInstanceOf(Outcome.Confident.class, down.outcome()).decision().value());
        assertEquals(before, meaning.calls(), "later stages are not asked");

        Cascade.Step<Priority> broken = cascade.step(State.of("login broken"));
        assertEquals(1, broken.stage());
        assertEquals(Priority.HIGH, assertInstanceOf(Outcome.Confident.class, broken.outcome()).decision().value());

        Cascade.Step<Priority> vague = cascade.step(State.of("hello"));
        assertEquals(1, vague.stage(), "uncertain: the last stage's outcome");
        var uncertain = assertInstanceOf(Outcome.Uncertain.class, vague.outcome());
        assertEquals(List.of(Decision.VALUE), uncertain.uncertainFields());
        assertEquals(Priority.NORMAL, uncertain.decision().value());
        assertSame(vague.outcome().getClass(), cascade.decide(State.of("hello")).getClass());
    }

    @Test
    void cascadeNeedsAStage() {
        assertThrows(IllegalArgumentException.class, () -> Cascade.of(List.of()));
    }
}
