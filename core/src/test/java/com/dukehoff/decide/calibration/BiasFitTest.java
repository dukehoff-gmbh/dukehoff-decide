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

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration.Labeled;
import com.dukehoff.decide.calibration.Calibration.Method;
import com.dukehoff.decide.calibration.CalibrationTest.Article;
import com.dukehoff.decide.calibration.CalibrationTest.Simulation;
import com.dukehoff.decide.calibration.CalibrationTest.Topic;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class BiasFitTest {

    /**
     * Labels are drawn from softmax(z), but the backend reports {@code 2 * z}, plus 1.5 on TECH for the topic:
     * overconfident and biased towards TECH. The opinion field is only overconfident.
     */
    static final class BiasedSimulation {
        final Map<String, double[]> topic = new HashMap<>();
        final Map<String, double[]> opinion = new HashMap<>();
        final List<Labeled<Article>> examples = new ArrayList<>();
        final ScriptedBackend backend;

        BiasedSimulation(int n, long seed) {
            Random rnd = new Random(seed);
            for (int i = 0; i < n; i++) {
                String id = "b" + i;
                double[] zt = Simulation.gaussian(rnd, 4, 1.5);
                double[] zo = Simulation.gaussian(rnd, 2, 1.5);
                Topic t = Topic.values()[Simulation.sample(rnd, Calibrator.softmax(zt, 1))];
                boolean o = Simulation.sample(rnd, Calibrator.softmax(zo, 1)) == 0;
                double[] reported = Simulation.scale(zt, 2);
                reported[Topic.TECH.ordinal()] += 1.5;
                topic.put(id, reported);
                opinion.put(id, Simulation.scale(zo, 2));
                examples.add(new Labeled<>(State.builder().put("id", id).build(), new Article(t, o)));
            }
            backend = new ScriptedBackend("biased-model", (state, field) -> {
                String id = state.attributes().get("id");
                return field.name().equals("topic") ? topic.get(id) : opinion.get(id);
            });
        }
    }

    @Test
    void correctsAModelThatFavoursOneOption() {
        BiasedSimulation sim = new BiasedSimulation(6000, 17);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);
        var train = sim.examples.subList(0, 4000);
        var holdout = sim.examples.subList(4000, 6000);

        var best = Calibration.fit(task, train, holdout);
        var temperatureOnly = Calibration.fit(task, train, holdout, Method.TEMPERATURE);

        var topic = best.fields().get(0);
        assertTrue(topic.biased(), best.toString());
        var bias = assertInstanceOf(Calibrator.Bias.class, topic.calibrator());
        assertEquals(2.0, bias.t(), 0.3, "recovers the overconfidence");
        double others = (bias.biasByLabel().get("SPORTS") + bias.biasByLabel().get("POLITICS")
                + bias.biasByLabel().get("CULTURE")) / 3;
        // the backend adds 1.5 to TECH's raw score; divided by t = 2 that is 0.75 too much
        assertEquals(-0.75, bias.biasByLabel().get("TECH") - others, 0.25, bias.toString());
        assertEquals(0, bias.biasByLabel().values().stream().mapToDouble(Double::doubleValue).sum(), 1e-6,
                "L2 on all biases centres them");

        var topicTemperature = temperatureOnly.fields().get(0);
        assertFalse(topicTemperature.biased());
        assertTrue(topic.nllAfter() < topic.nllTemperatureOnly());
        assertEquals(topicTemperature.nllAfter(), topic.nllTemperatureOnly(), 1e-12);
        assertTrue(topic.accuracy() > topicTemperature.accuracy() + 0.02,
                "bias changes wrong winners: " + topic.accuracy() + " vs " + topicTemperature.accuracy());
        assertTrue(topic.eceAfter() < 0.04, "ECE " + topic.eceAfter());
        assertTrue(best.toString().contains("topic: bias chosen"), best.toString());
    }

    @Test
    void findsNoBiasWhereThereIsNone() {
        Simulation sim = new Simulation(5000, 3.0, 23);
        var raw = new ArrayList<double[]>();
        int[] targets = new int[sim.examples.size()];
        for (int i = 0; i < targets.length; i++) {
            raw.add(sim.topicLogits.get("a" + i));
            targets[i] = sim.examples.get(i).expected().topic().ordinal();
        }
        var bias = BiasFit.fit(raw, targets, List.of("SPORTS", "POLITICS", "TECH", "CULTURE")).orElseThrow();
        assertEquals(3.0, bias.t(), 0.3);
        bias.biasByLabel().values().forEach(b -> assertEquals(0, b, 0.15, bias.toString()));
    }

    @Test
    void temperatureWithoutHoldoutOrWhenAsked() {
        BiasedSimulation sim = new BiasedSimulation(2000, 3);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);
        assertTrue(Calibration.fit(task, sim.examples).fields().stream().noneMatch(Calibration.FieldReport::biased),
                "no holdout: nothing to compare on, so temperature");
        assertTrue(Calibration.fit(task, sim.examples.subList(0, 1000), sim.examples.subList(1000, 2000), Method.TEMPERATURE)
                .fields().stream().noneMatch(Calibration.FieldReport::biased));
    }

    @Test
    void storedBiasIsAppliedInOptionOrder(@TempDir Path dir) {
        BiasedSimulation sim = new BiasedSimulation(3000, 5);
        CalibrationStore store = CalibrationStore.directory(dir);
        DecisionTask<Article> raw = Decider.builder().backend(sim.backend).build().task(Article.class);
        var report = Calibration.fit(raw, sim.examples.subList(0, 2000), sim.examples.subList(2000, 3000));
        var fitted = assertInstanceOf(Calibrator.Bias.class, report.profile().byField().get("topic"));

        // store the biases in reverse order: loading must still map them to the right options
        Map<String, Double> reversed = new LinkedHashMap<>();
        new ArrayList<>(fitted.biasByLabel().keySet()).reversed().forEach(l -> reversed.put(l, fitted.biasByLabel().get(l)));
        Map<String, Calibrator> byField = new LinkedHashMap<>(report.profile().byField());
        byField.put("topic", new Calibrator.Bias(fitted.t(), reversed));
        store.save(new Calibration.Profile(report.profile().taskId(), report.profile().modelId(), byField));

        DecisionTask<Article> calibrated = Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build().task(Article.class);
        State s = sim.examples.get(2500).state();
        Decision<Article> d = calibrated.decide(s);
        assertTrue(d.calibrated());
        assertArrayEquals(fitted.apply(sim.topic.get("b2500")), d.field("topic").probabilities(), 1e-12);
    }

    @Test
    void profileForOtherOptionsIsIgnored() {
        CalibrationStore store = CalibrationStore.inMemory();
        Map<String, Double> oldOptions = new LinkedHashMap<>();
        for (String l : List.of("SPORTS", "POLITICS", "TECH", "ARTS")) oldOptions.put(l, 0.1);
        store.save(new Calibration.Profile(Article.class.getName(), "biased-model", Map.of(
                "topic", new Calibrator.Bias(2, oldOptions), "opinion", Calibrator.temperature(2))));
        BiasedSimulation sim = new BiasedSimulation(10, 1);

        DecisionTask<Article> lenient = Decider.builder().backend(sim.backend).calibration(store).build().task(Article.class);
        Decision<Article> d = lenient.decide(sim.examples.getFirst().state());
        assertFalse(d.calibrated(), "a profile for other enum constants is treated as missing");
        assertArrayEquals(Calibrator.softmax(sim.opinion.get("b0"), 1), d.field("opinion").probabilities(), 1e-12,
                "and not used for the other fields either");

        var e = assertThrows(DecideException.class, () -> Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build().task(Article.class));
        assertTrue(e.getMessage().contains("other options"), e.getMessage());
    }

    @Test
    void biasApplyAndRebinding() {
        var bias = new Calibrator.Bias(2, new LinkedHashMap<>(Map.of("yes", 0.5)));
        assertThrows(IllegalArgumentException.class, () -> bias.apply(new double[] {0, 0}));
        assertThrows(IllegalArgumentException.class, () -> new Calibrator.Bias(0, Map.of("a", 0.0)));
        assertThrows(IllegalArgumentException.class, () -> new Calibrator.Bias(1, Map.of("a", Double.NaN)));

        Map<String, Double> b = new LinkedHashMap<>();
        b.put("no", -1.0);
        b.put("yes", 1.0);
        var yesNo = new Calibrator.Bias(1, b).forOptions(List.of("yes", "no")).orElseThrow();
        double[] p = yesNo.apply(new double[] {0, 0});
        assertEquals(Math.exp(1) / (Math.exp(1) + Math.exp(-1)), p[0], 1e-12, "bias of 'yes' applies to option 0");
        assertTrue(new Calibrator.Bias(1, b).forOptions(List.of("yes", "maybe")).isEmpty());

        double[] withMissing = yesNo.apply(new double[] {Double.NEGATIVE_INFINITY, 0});
        assertEquals(0.0, withMissing[0], 0);
        assertEquals(1.0, withMissing[1], 0);
    }
}
