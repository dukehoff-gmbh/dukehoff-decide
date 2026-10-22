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

import com.dukehoff.decide.Fixtures.Triage;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.TaskSchema;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LabelPromptTest {

    final TaskSchema schema = Decider.builder()
            .backend(new ScriptedBackend("scripted", (s, f) -> ScriptedBackend.uniform(f)))
            .build().task(Triage.class).schema();

    final State state = State.builder().put("channel", "email").text("Server down since 3am!").build();

    @Test
    void labelsOptionsWithLettersInOptionOrder() {
        LabelPrompt p = LabelPrompt.of(schema, schema.field("priority"), state);
        assertEquals(List.of("A", "B", "C", "D"), p.labels());
        assertTrue(p.system().contains("Triage an incoming customer support ticket"), p.system());
        assertTrue(p.user().contains("channel: email\n\nServer down since 3am!"), p.user());
        assertTrue(p.user().contains("Question: How urgent is this ticket?"), p.user());
        assertTrue(p.user().contains("A) BLOCKER: Production is down or data is being lost\n"), p.user());
        assertTrue(p.user().contains("D) LOW: Question, cosmetic issue or feature wish\n"), p.user());
    }

    @Test
    void rendersBooleansAndScales() {
        LabelPrompt churn = LabelPrompt.of(schema, schema.field("churnRisk"), state);
        assertTrue(churn.user().contains("A) yes\nB) no\n"), churn.user());

        LabelPrompt anger = LabelPrompt.of(schema, schema.field("anger"), state);
        assertTrue(anger.user().contains("A) 0 = calm\nB) 0.250\n"), anger.user());
        assertTrue(anger.user().contains("E) 1 = furious\n"), anger.user());
    }

    @Test
    void mapsTokensBackToOptionOrder() {
        LabelPrompt p = LabelPrompt.of(schema, schema.field("priority"), state);
        double[] s = p.scores(Map.of("B", -2.0, "A", -0.1, "The", -3.0), -30);
        assertArrayEquals(new double[] {-0.1, -2.0, -30, -30}, s, 1e-12);
    }

    @Test
    void mergesTokenVariantsOfTheSameLabel() {
        LabelPrompt p = LabelPrompt.of(schema, schema.field("churnRisk"), state);
        double[] s = p.scores(Map.of("A", Math.log(0.5), " A", Math.log(0.2), "A)", Math.log(0.1)), -30);
        assertEquals(Math.log(0.8), s[0], 1e-12);
        assertEquals(-30, s[1]);
    }
}
