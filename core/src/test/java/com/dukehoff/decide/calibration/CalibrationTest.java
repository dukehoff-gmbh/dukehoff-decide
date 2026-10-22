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
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration.Labeled;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class CalibrationTest {

    enum Topic { SPORTS, POLITICS, TECH, CULTURE }

    record Article(@Question("What is the article about?") Topic topic,
                   @Question("Is it an opinion piece?") boolean opinion) {}

    /**
     * Simulates an overconfident model: the true label is drawn from softmax(z), i.e. z would be
     * perfectly calibrated, but the backend reports {@code overconfidence * z}.
     */
    static final class Simulation {
        final Map<String, double[]> topicLogits = new HashMap<>();
        final Map<String, double[]> opinionLogits = new HashMap<>();
        final List<Labeled<Article>> examples = new ArrayList<>();
        final ScriptedBackend backend;

        Simulation(int n, double overconfidence, long seed) {
            Random rnd = new Random(seed);
            for (int i = 0; i < n; i++) {
                String id = "a" + i;
                double[] zt = gaussian(rnd, 4, 1.5);
                double[] zo = gaussian(rnd, 2, 1.5);
                Topic topic = Topic.values()[sample(rnd, Calibrator.softmax(zt, 1))];
                boolean opinion = sample(rnd, Calibrator.softmax(zo, 1)) == 0;
                topicLogits.put(id, scale(zt, overconfidence));
                opinionLogits.put(id, scale(zo, overconfidence));
                examples.add(new Labeled<>(State.builder().put("id", id).build(), new Article(topic, opinion)));
            }
            backend = new ScriptedBackend("sim-model", (state, field) -> {
                String id = state.attributes().get("id");
                return field.name().equals("topic") ? topicLogits.get(id) : opinionLogits.get(id);
            });
        }

        static double[] gaussian(Random rnd, int k, double sd) {
            double[] z = new double[k];
            for (int i = 0; i < k; i++) z[i] = rnd.nextGaussian() * sd;
            return z;
        }

        static double[] scale(double[] z, double f) {
            double[] out = new double[z.length];
            for (int i = 0; i < z.length; i++) out[i] = z[i] * f;
            return out;
        }

        static int sample(Random rnd, double[] p) {
            double u = rnd.nextDouble();
            for (int i = 0; i < p.length; i++) {
                u -= p[i];
                if (u <= 0) return i;
            }
            return p.length - 1;
        }
    }

    @Test
    void temperatureFitRecoversOverconfidence() {
        Simulation sim = new Simulation(6000, 3.0, 42);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);

        var report = Calibration.fit(task, sim.examples.subList(0, 4000), sim.examples.subList(4000, 6000));
        assertTrue(report.evaluatedOnHoldout());
        for (var f : report.fields()) {
            assertEquals(3.0, f.temperature(), 0.35, f.field() + " temperature");
            assertTrue(f.eceAfter() < f.eceBefore() / 2, f.field() + " ECE should drop: " + f);
            assertTrue(f.eceAfter() < 0.04, f.field() + " ECE after: " + f.eceAfter());
            assertTrue(f.nllAfter() < f.nllBefore(), f.field() + " NLL should drop");
            assertEquals(2000, f.examples());
        }
        assertEquals("sim-model", report.profile().modelId());
        assertEquals(Article.class.getName(), report.profile().taskId());
        assertTrue(report.toString().contains("ECE before"), report.toString());
    }

    @Test
    void wellCalibratedModelKeepsTemperatureNearOne() {
        Simulation sim = new Simulation(4000, 1.0, 7);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);
        var report = Calibration.fit(task, sim.examples);
        assertFalse(report.evaluatedOnHoldout());
        for (var f : report.fields()) assertEquals(1.0, f.temperature(), 0.15, f.field());
    }

    @Test
    void deciderUsesStoredProfile() throws Exception {
        Simulation sim = new Simulation(3000, 3.0, 1);
        Path dir = Files.createTempDirectory("dukehoff-decide-cal");
        CalibrationStore store = CalibrationStore.directory(dir);

        Decider decider = Decider.builder().backend(sim.backend).calibration(store).build();
        DecisionTask<Article> raw = decider.task(Article.class);
        store.save(Calibration.fit(raw, sim.examples).profile());

        Decider calibratedDecider = Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build();
        DecisionTask<Article> calibrated = calibratedDecider.task(Article.class);

        State s = sim.examples.getFirst().state();
        Decision<Article> before = raw.decide(s);
        Decision<Article> after = calibrated.decide(s);
        assertFalse(before.calibrated());
        assertTrue(after.calibrated());
        assertEquals(before.value(), after.value(), "temperature scaling never changes the winner");
        assertTrue(after.field("topic").confidence() <= before.field("topic").confidence());

        try (var files = Files.list(dir)) {
            Path file = files.findFirst().orElseThrow();
            String content = Files.readString(file);
            assertTrue(content.contains("field.topic.temperature="), content);
            assertTrue(content.contains("modelId=sim-model"), content);
        }
    }

    @Test
    void storeSeparatesModels() throws Exception {
        CalibrationStore store = CalibrationStore.directory(Files.createTempDirectory("dukehoff-decide-cal"));
        store.save(new Calibration.Profile("t", "model:a", Map.of("f", Calibrator.temperature(2))));
        assertTrue(store.load("t", "model:a").isPresent());
        assertTrue(store.load("t", "model:b").isEmpty());
        assertTrue(store.load("t", "model_a").isEmpty(), "sanitised name collision must not match");
        var t = assertInstanceOf(Calibrator.Temperature.class, store.load("t", "model:a").get().byField().get("f"));
        assertEquals(2.0, t.t(), 0);

        CalibrationStore mem = CalibrationStore.inMemory();
        mem.save(new Calibration.Profile("t", "m", Map.of()));
        assertTrue(mem.load("t", "m").isPresent());
    }

    @Test
    void classpathStoreReadsWhatDirectoryStoreWrote() throws Exception {
        Path root = Files.createTempDirectory("dukehoff-decide-cp");
        CalibrationStore.directory(root.resolve("decide/calibration"))
                .save(new Calibration.Profile("com.acme.Triage", "qwen:7b", Map.of("priority", Calibrator.temperature(1.8))));

        try (var loader = new java.net.URLClassLoader(new java.net.URL[] {root.toUri().toURL()}, null)) {
            for (String dir : List.of("decide/calibration", "/decide/calibration/")) {
                CalibrationStore cp = CalibrationStore.classpath(dir, loader);
                var p = cp.load("com.acme.Triage", "qwen:7b").orElseThrow();
                assertEquals(1.8, assertInstanceOf(Calibrator.Temperature.class, p.byField().get("priority")).t(), 0);
                assertTrue(cp.load("com.acme.Triage", "qwen:1.5b").isEmpty());
                assertThrows(UnsupportedOperationException.class, () -> cp.save(p));
            }
        }
    }

    @Test
    void softmaxAndTemperature() {
        double[] p = Calibrator.softmax(new double[]{Math.log(0.2), Math.log(0.8)}, 1);
        assertEquals(0.8, p[1], 1e-12);
        double[] soft = Calibrator.temperature(1e6).apply(new double[]{0, 10});
        assertEquals(0.5, soft[0], 1e-4);
        double[] withZero = Calibrator.softmax(new double[]{Double.NEGATIVE_INFINITY, 0}, 2);
        assertEquals(0.0, withZero[0], 0);
        assertThrows(IllegalArgumentException.class, () -> Calibrator.temperature(0));
    }

    @Test
    void metrics() {
        List<double[]> probs = List.of(new double[]{0.9, 0.1}, new double[]{0.9, 0.1}, new double[]{0.6, 0.4});
        int[] targets = {0, 1, 0};
        assertEquals(2.0 / 3, Metrics.accuracy(probs, targets), 1e-12);
        // bin 0.9: conf 0.9 x2, correct 1 -> |1.8 - 1| / 3; bin 0.6: |0.6 - 1| / 3
        assertEquals((0.8 + 0.4) / 3, Metrics.ece(probs, targets), 1e-12);
        assertEquals(-(Math.log(0.9) + Math.log(0.1) + Math.log(0.6)) / 3, Metrics.nll(probs, targets), 1e-12);

        List<double[]> perfect = List.of(new double[]{1, 0}, new double[]{0, 1});
        assertEquals(0.0, Metrics.ece(perfect, new int[]{0, 1}), 1e-12);
    }
}
