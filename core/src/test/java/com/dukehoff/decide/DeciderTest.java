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

import com.dukehoff.decide.Fixtures.*;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.dukehoff.decide.testing.ScriptedBackend.probabilities;
import static org.junit.jupiter.api.Assertions.*;

class DeciderTest {

    /** "down" tickets: sure BLOCKER, PLATFORM, torn on churn, very angry. Everything else: LOW, unsure. */
    static final ScriptedBackend BACKEND = new ScriptedBackend("scripted", (state, field) -> {
        boolean down = state.text().contains("down");
        return switch (field.name()) {
            case "priority" -> down ? probabilities(0.90, 0.06, 0.03, 0.01) : probabilities(0.1, 0.2, 0.3, 0.4);
            case "team" -> down ? probabilities(0.05, 0.85, 0.05, 0.05) : probabilities(0.25, 0.25, 0.25, 0.25);
            case "churnRisk" -> down ? probabilities(0.55, 0.45) : probabilities(0.02, 0.98);
            case "anger" -> down ? probabilities(0.0, 0.0, 0.0, 0.2, 0.8) : probabilities(0.2, 0.2, 0.2, 0.2, 0.2);
            case "stars" -> probabilities(0.0, 0.0, 0.3, 0.7, 0.0);
            default -> ScriptedBackend.uniform(field);
        };
    });

    final Decider decider = Decider.builder().backend(BACKEND).build();

    @Test
    void decidesATypedRecord() {
        Decision<Triage> d = decider.task(Triage.class).decide(State.of("Server down since 3am!"));
        Triage t = d.value();
        assertEquals(Priority.BLOCKER, t.priority());
        assertEquals(Team.PLATFORM, t.team());
        assertTrue(t.churnRisk());
        assertEquals(0.95, t.anger(), 1e-9); // 0.2 * 0.75 + 0.8 * 1.0
        assertFalse(d.calibrated());

        Map<Priority, Double> dist = d.distribution("priority");
        assertEquals(List.of(Priority.BLOCKER, Priority.HIGH, Priority.NORMAL, Priority.LOW), List.copyOf(dist.keySet()));
        assertEquals(0.90, dist.get(Priority.BLOCKER), 1e-9);

        // weakest field is churnRisk at 0.55
        assertEquals(0.55, d.confidence(), 1e-9);
        var churn = assertInstanceOf(FieldResult.Bool.class, d.field("churnRisk"));
        assertEquals(0.55, churn.probabilityTrue(), 1e-9);
    }

    @Test
    void scoreReportsSpreadAndRoundsIntegers() {
        Decision<Stars> d = decider.task(Stars.class).decide(State.of("Pretty good"));
        assertEquals(4, d.value().stars()); // 0.3 * 3 + 0.7 * 4 = 3.7
        var score = assertInstanceOf(FieldResult.Score.class, d.field("stars"));
        assertEquals(3.7, score.value(), 1e-9);
        double sd = Math.sqrt(0.3 * 0.49 + 0.7 * 0.09);
        assertEquals(sd / 2, score.spread(), 1e-9);
        assertEquals(1 - sd / 2, score.confidence(), 1e-9);
    }

    @Test
    void gatingSeparatesConfidentFromUncertain() {
        var gated = decider.task(Triage.class).requireConfidence(Map.of("priority", 0.8, "team", 0.8));
        assertInstanceOf(Outcome.Confident.class, gated.decide(State.of("Server down")));

        Outcome<Triage> unsure = gated.decide(State.of("How do I change my avatar?"));
        switch (unsure) {
            case Outcome.Confident<Triage> c -> fail("expected uncertain");
            case Outcome.Uncertain<Triage>(var d, var fields) -> {
                assertEquals(List.of("priority", "team"), fields);
                assertEquals(Priority.LOW, d.value().priority());
            }
        }

        var strict = decider.task(Triage.class).requireConfidence(0.6);
        var u = assertInstanceOf(Outcome.Uncertain.class, strict.decide(State.of("Server down")));
        assertEquals(List.of("churnRisk"), u.uncertainFields());
    }

    @Test
    void gatingRejectsUnknownFieldsAndBadThresholds() {
        var task = decider.task(Triage.class);
        assertThrows(IllegalArgumentException.class, () -> task.requireConfidence(Map.of("nope", 0.5)));
        assertThrows(IllegalArgumentException.class, () -> task.requireConfidence(1.5));
    }

    @Test
    void enumAndBooleanTasks() {
        Decision<Priority> p = decider.task(Priority.class, "How urgent?").decide(State.of("down"));
        assertEquals(Priority.BLOCKER, p.value());
        Map<Priority, Double> dist = p.distribution();
        assertEquals(4, dist.size());

        ScriptedBackend yes = new ScriptedBackend("yes", (s, f) -> probabilities(0.8, 0.2));
        Decision<Boolean> b = Decider.builder().backend(yes).build()
                .task(Boolean.class, "Is this spam?").decide(State.of("WIN A PRIZE"));
        assertTrue(b.value());
        assertEquals(0.8, b.confidence(), 1e-9);
    }

    @Test
    void batchAndAsyncKeepOrder() throws Exception {
        var task = decider.task(Triage.class);
        List<Decision<Triage>> all = task.decideAll(List.of(State.of("down"), State.of("hello"), State.of("down")));
        assertEquals(List.of(Priority.BLOCKER, Priority.LOW, Priority.BLOCKER),
                all.stream().map(d -> d.value().priority()).toList());
        assertEquals(Priority.BLOCKER, task.decideAsync(State.of("down")).get(5, TimeUnit.SECONDS).value().priority());
    }

    @Test
    void invalidBackendOutputIsRejected() {
        ScriptedBackend tooShort = new ScriptedBackend("short", (s, f) -> new double[]{0});
        var e = assertThrows(DecideException.class,
                () -> Decider.builder().backend(tooShort).build().task(Triage.class).decide(State.of("x")));
        assertTrue(e.getMessage().contains("expected 4"), e.getMessage());

        ScriptedBackend nan = new ScriptedBackend("nan", (s, f) -> new double[]{Double.NaN, 0});
        assertThrows(DecideException.class,
                () -> Decider.builder().backend(nan).build().task(Boolean.class, "Spam?").decide(State.of("x")));

        ScoringBackend missing = new ScoringBackend() {
            public String modelId() { return "missing"; }
            public Map<String, RawScores> score(ScoringRequest r) { return Map.of(); }
        };
        assertThrows(DecideException.class,
                () -> Decider.builder().backend(missing).build().task(Triage.class).decide(State.of("x")));

        ScoringBackend failing = new ScoringBackend() {
            public String modelId() { return "failing"; }
            public Map<String, RawScores> score(ScoringRequest r) { throw new IllegalStateException("boom"); }
        };
        var wrapped = assertThrows(DecideException.class,
                () -> Decider.builder().backend(failing).build().task(Triage.class).decide(State.of("x")));
        assertInstanceOf(IllegalStateException.class, wrapped.getCause());
    }

    @Test
    void recordValidationFailuresSurfaceAsDecideException() {
        ScriptedBackend spam = new ScriptedBackend("spam", (s, f) -> probabilities(0.9, 0.1));
        var e = assertThrows(DecideException.class,
                () -> Decider.builder().backend(spam).build().task(Validated.class).decide(State.of("x")));
        assertInstanceOf(IllegalArgumentException.class, e.getCause());
    }

    @Test
    void requireCalibrationFailsWithoutProfile() {
        Decider strict = Decider.builder().backend(BACKEND).requireCalibration(true).build();
        var e = assertThrows(DecideException.class, () -> strict.task(Triage.class));
        assertTrue(e.getMessage().contains("Calibration.fit"), e.getMessage());
    }

    @Test
    void stateRendersAttributesBeforeText() {
        State s = State.builder().put("subject", "Login broken").put("plan", "enterprise").text("Since today.").build();
        assertEquals("subject: Login broken\nplan: enterprise\n\nSince today.", s.render());
        assertEquals("Only text", State.of("Only text").render());
    }

    @Test
    void backendSeesSchemaAndState() {
        var seen = new java.util.ArrayList<String>();
        ScriptedBackend spy = new ScriptedBackend("spy", (s, f) -> {
            seen.add(f.name() + ":" + f.question());
            return ScriptedBackend.uniform(f);
        });
        Decider.builder().backend(spy).build().task(Triage.class).decide(State.of("x"));
        assertEquals(4, seen.size());
        assertEquals("priority:How urgent is this ticket?", seen.getFirst());
        assertEquals(1, spy.calls());
        assertInstanceOf(FieldSpec.ScoreSpec.class,
                Decider.builder().backend(spy).build().task(Triage.class).schema().field("anger"));
    }
}
