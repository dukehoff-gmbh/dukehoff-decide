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
package com.dukehoff.decide.calibration;

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Outcome;
import com.dukehoff.decide.calibration.Calibration.Labeled;
import com.dukehoff.decide.calibration.CalibrationTest.Article;
import com.dukehoff.decide.calibration.CalibrationTest.Simulation;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class CoverageTest {

    @Test
    void countsDecidedAndCorrectPerThreshold() {
        double[] confidence = {0.95, 0.9, 0.8, 0.6, 0.55, 0.4};
        boolean[] correct = {true, true, false, true, false, false};
        Coverage c = Coverage.of(confidence, correct);

        assertEquals(50, c.points().size());
        assertEquals(new Coverage.Point(0.5, 6, 5, 3), c.at(0.5));
        assertEquals(new Coverage.Point(0.8, 6, 3, 2), c.at(0.8), "0.8 >= 0.8 counts as decided, like the gate");
        assertEquals(new Coverage.Point(0.99, 6, 0, 0), c.at(0.99));
        assertEquals(0.5, c.at(0.8).coverage(), 1e-12);
        assertEquals(2.0 / 3, c.at(0.8).accuracy(), 1e-12);
        assertTrue(Double.isNaN(c.at(0.99).accuracy()));
        assertThrows(IllegalArgumentException.class, () -> c.at(0.555));

        assertEquals(0.81, c.thresholdFor(1.0).orElseThrow(), 1e-12, "lowest threshold with 100%: only 0.95 and 0.9 left");
        assertEquals(0.5, c.thresholdFor(0.6).orElseThrow(), 1e-12);
        assertTrue(Coverage.of(new double[] {0.9}, new boolean[] {false}).thresholdFor(0.5).isEmpty());
    }

    /** The curve in the report must predict exactly what requireConfidence does at runtime. */
    @Test
    void curveMatchesRuntimeGating() {
        Simulation sim = new Simulation(3000, 3.0, 11);
        List<Labeled<Article>> train = sim.examples.subList(0, 2000);
        List<Labeled<Article>> holdout = sim.examples.subList(2000, 3000);
        CalibrationStore store = CalibrationStore.inMemory();
        DecisionTask<Article> raw = Decider.builder().backend(sim.backend).calibration(store).build().task(Article.class);
        var report = Calibration.fit(raw, train, holdout);
        store.save(report.profile());
        DecisionTask<Article> calibrated = Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build().task(Article.class);

        for (double t : List.of(0.5, 0.6, 0.73, 0.8, 0.9, 0.99)) {
            assertGateMatches(report.coverage(Calibration.WHOLE_RECORD).at(t), calibrated.requireConfidence(t),
                    holdout, a -> a);
            assertGateMatches(report.coverage("topic").at(t), calibrated.requireConfidence(Map.of("topic", t)),
                    holdout, Article::topic);
            assertGateMatches(report.wholeRecordBefore().at(t), raw.requireConfidence(t), holdout, a -> a);
            assertGateMatches(report.fields().get(1).coverageBefore().at(t),
                    raw.requireConfidence(Map.of("opinion", t)), holdout, Article::opinion);
        }
    }

    private static void assertGateMatches(Coverage.Point expected, DecisionTask.Gated<Article> gate,
                                          List<Labeled<Article>> examples, Function<Article, Object> target) {
        int decided = 0, correct = 0;
        for (Labeled<Article> e : examples) {
            if (gate.decide(e.state()) instanceof Outcome.Confident<Article>(var d)) {
                decided++;
                if (target.apply(d.value()).equals(target.apply(e.expected()))) correct++;
            }
        }
        assertEquals(expected.decided(), decided, "decided at " + expected.threshold());
        assertEquals(expected.correct(), correct, "correct at " + expected.threshold());
    }

    @Test
    void calibrationMakesTheCurveHonest() {
        Simulation sim = new Simulation(4000, 3.0, 5);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);
        var report = Calibration.fit(task, sim.examples.subList(0, 3000), sim.examples.subList(3000, 4000));

        // the overconfident raw model decides much more at 0.9 than it should, and is wrong more often
        Coverage.Point before = report.fields().getFirst().coverageBefore().at(0.9);
        Coverage.Point after = report.coverage("topic").at(0.9);
        assertTrue(before.coverage() > after.coverage(), before + " vs " + after);
        assertTrue(before.accuracy() < 0.9, "raw: " + before);
        assertTrue(after.accuracy() >= 0.88, "calibrated: " + after);

        double t = report.thresholdFor("topic", 0.9).orElseThrow();
        assertTrue(report.coverage("topic").at(t).accuracy() >= 0.9);
        assertThrows(IllegalArgumentException.class, () -> report.coverage("nope"));

        String text = report.toString();
        assertTrue(text.contains("decided automatically after calibration"), text);
        assertTrue(text.contains(Calibration.WHOLE_RECORD), text);
    }
}
