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
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration.Labeled;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class ConfusionTest {

    static final List<String> ABCD = List.of("A", "B", "C", "D");

    /** rows: expected, columns: predicted */
    static Confusion matrix(int[][] counts) {
        return new Confusion(ABCD, counts);
    }

    @Test
    void countsPrecisionRecallAndTopConfusions() {
        Confusion c = matrix(new int[][] {
                {8, 0, 2, 0},
                {1, 5, 4, 0},
                {0, 0, 9, 1},
                {0, 0, 0, 0}});
        assertEquals(30, c.examples());
        assertEquals(8, c.errors());
        assertEquals(10, c.support(1));
        assertEquals(8.0 / 9, c.precision(0), 1e-12);
        assertEquals(9.0 / 15, c.precision(2), 1e-12);
        assertEquals(0.5, c.recall(1), 1e-12);
        assertTrue(Double.isNaN(c.recall(3)), "no examples of D");
        assertEquals(0.0, c.precision(3), 0, "D predicted once, wrongly");
        assertEquals(List.of(new Confusion.Cell("B", "C", 4), new Confusion.Cell("A", "C", 2)), c.topConfusions(2));
        assertEquals(4, c.topConfusions(10).size());
    }

    @Test
    void ofUsesArgmax() {
        Confusion c = Confusion.of(List.of("yes", "no"),
                List.of(new double[] {0.9, 0.1}, new double[] {0.3, 0.7}, new double[] {0.6, 0.4}), new int[] {0, 0, 1});
        assertArrayEquals(new int[] {1, 1}, c.counts()[0]);
        assertArrayEquals(new int[] {1, 0}, c.counts()[1]);
    }

    @Test
    void findsAnAttractor() {
        Confusion c = matrix(new int[][] {
                {10, 0, 5, 0},
                {0, 10, 4, 1},
                {1, 0, 10, 0},
                {0, 0, 3, 10}});
        var a = c.attractor().orElseThrow();
        assertEquals("C", a.label());
        assertEquals(12, a.errors());
        assertEquals(12.0 / 14, a.share(), 1e-12);
    }

    @Test
    void noAttractorWhenErrorsAreSpreadFewOrBinary() {
        Confusion spread = matrix(new int[][] {
                {10, 2, 2, 2},
                {2, 10, 2, 2},
                {2, 2, 10, 2},
                {2, 2, 2, 10}});
        assertTrue(spread.attractor().isEmpty());

        Confusion few = matrix(new int[][] {
                {10, 0, 4, 0},
                {0, 10, 0, 0},
                {0, 0, 10, 0},
                {0, 0, 0, 10}});
        assertTrue(few.attractor().isEmpty(), "4 errors are not enough evidence");

        Confusion binary = new Confusion(List.of("yes", "no"), new int[][] {{10, 0}, {20, 10}});
        assertTrue(binary.attractor().isEmpty());
    }

    enum Topic { SPORTS, POLITICS, TECH, CULTURE }

    record Article(@Question("What is the article about?") Topic topic,
                   @Question("Is it an opinion piece?") boolean opinion) {}

    /**
     * topic: right about half the time, otherwise always TECH, i.e. TECH is an attractor.
     * opinion: confident and unrelated to the label, so the temperature runs into the limit.
     */
    @Test
    void reportShowsConfusionsAndWarnings() {
        Random rnd = new Random(3);
        List<Labeled<Article>> examples = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            Topic topic = Topic.values()[rnd.nextInt(4)];
            examples.add(new Labeled<>(State.builder().put("id", i).put("topic", topic).build(),
                    new Article(topic, rnd.nextBoolean())));
        }
        Random noise = new Random(4);
        var backend = new ScriptedBackend("sim", (state, field) -> {
            if (field.name().equals("opinion")) return noise.nextBoolean() ? new double[] {0, -20} : new double[] {-20, 0};
            int truth = Topic.valueOf(state.attributes().get("topic")).ordinal();
            int id = Integer.parseInt(state.attributes().get("id"));
            int predicted = id % 2 == 0 ? truth : Topic.TECH.ordinal();
            double[] lp = {-4, -4, -4, -4};
            lp[predicted] = -0.1;
            return lp;
        });
        DecisionTask<Article> task = Decider.builder().backend(backend).build().task(Article.class);

        var report = Calibration.fit(task, examples);
        var topic = report.fields().get(0);
        assertEquals("TECH", topic.confusion().attractor().orElseThrow().label());
        assertFalse(topic.temperatureAtLimit());
        var opinion = report.fields().get(1);
        assertTrue(opinion.temperatureAtLimit(), "T = " + opinion.temperature());

        String text = report.toString();
        assertTrue(text.contains("topic top confusions (expected -> predicted): "), text);
        assertTrue(text.contains("! topic: option TECH receives"), text);
        assertTrue(text.contains("! opinion: temperature ") && text.contains("is at the limit"), text);
    }
}
