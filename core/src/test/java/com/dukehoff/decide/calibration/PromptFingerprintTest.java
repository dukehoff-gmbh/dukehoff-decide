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
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.calibration.CalibrationTest.Article;
import com.dukehoff.decide.calibration.CalibrationTest.Simulation;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.TaskSchema;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class PromptFingerprintTest {

    @Question("How urgent is it?")
    enum Urgency { @Option("Production is down") HIGH, @Option("Can wait") LOW }

    @Question("How urgent is it?")
    enum UrgencyReordered { @Option("Can wait") LOW, @Option("Production is down") HIGH }

    @Question("How urgent is it?")
    enum UrgencyReworded { @Option("Production or data loss") HIGH, @Option("Can wait") LOW }

    @Question("How pressing is it?")
    enum UrgencyRequestioned { @Option("Production is down") HIGH, @Option("Can wait") LOW }

    static final Decider DECIDER = Decider.builder()
            .backend(new ScriptedBackend("m", (s, f) -> ScriptedBackend.uniform(f))).build();

    static String fingerprint(Class<?> type, String instructions) {
        TaskSchema schema = DECIDER.task(type, instructions).schema();
        return LabelPrompt.fingerprint(schema, schema.fields().getFirst());
    }

    @Test
    void changesWithEverythingThatShapesThePrompt() {
        String base = fingerprint(Urgency.class, "");
        assertEquals(64, base.length());
        assertEquals(base, fingerprint(Urgency.class, ""), "deterministic");
        assertNotEquals(base, fingerprint(UrgencyReordered.class, ""), "option order");
        assertNotEquals(base, fingerprint(UrgencyReworded.class, ""), "option description");
        assertNotEquals(base, fingerprint(UrgencyRequestioned.class, ""), "question");
        assertNotEquals(base, fingerprint(Urgency.class, "Tickets are in German."), "instructions");
    }

    @Test
    void fitStoresFingerprintsAndTheStoreKeepsThem(@TempDir Path dir) throws Exception {
        Simulation sim = new Simulation(500, 2.0, 3);
        DecisionTask<Article> task = Decider.builder().backend(sim.backend).build().task(Article.class);
        var profile = Calibration.fit(task, sim.examples).profile();
        TaskSchema schema = task.schema();
        assertEquals(LabelPrompt.fingerprint(schema, schema.field("topic")), profile.promptHashByField().get("topic"));
        assertEquals(LabelPrompt.fingerprint(schema, schema.field("opinion")), profile.promptHashByField().get("opinion"));

        CalibrationStore store = CalibrationStore.directory(dir);
        store.save(profile);
        try (var files = Files.list(dir)) {
            assertTrue(Files.readString(files.findFirst().orElseThrow()).contains("field.topic.promptHash="));
        }
        assertEquals(profile.promptHashByField(),
                store.load(profile.taskId(), profile.modelId()).orElseThrow().promptHashByField());
    }

    @Test
    void changedPromptWarnsOrFails() {
        Simulation sim = new Simulation(500, 2.0, 4);
        DecisionTask<Article> raw = Decider.builder().backend(sim.backend).build().task(Article.class);
        var fitted = Calibration.fit(raw, sim.examples).profile();
        Map<String, String> hashes = new LinkedHashMap<>(fitted.promptHashByField());
        hashes.put("topic", "0".repeat(64)); // as if @Question or @Option texts had changed since fitting
        CalibrationStore store = CalibrationStore.inMemory();
        store.save(new Calibration.Profile(fitted.taskId(), fitted.modelId(), fitted.byField(), hashes));

        DecisionTask<Article> lenient = Decider.builder().backend(sim.backend).calibration(store).build().task(Article.class);
        assertTrue(lenient.decide(sim.examples.getFirst().state()).calibrated(), "stale but usable: warned, still used");

        var e = assertThrows(DecideException.class, () -> Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build().task(Article.class));
        assertTrue(e.getMessage().contains("different prompt of [topic]"), e.getMessage());
    }

    @Test
    void profilesWithoutFingerprintAreNotChecked() {
        Simulation sim = new Simulation(500, 2.0, 5);
        DecisionTask<Article> raw = Decider.builder().backend(sim.backend).build().task(Article.class);
        var fitted = Calibration.fit(raw, sim.examples).profile();
        CalibrationStore store = CalibrationStore.inMemory();
        store.save(new Calibration.Profile(fitted.taskId(), fitted.modelId(), fitted.byField()));

        DecisionTask<Article> strict = Decider.builder().backend(sim.backend).calibration(store)
                .requireCalibration(true).build().task(Article.class);
        assertTrue(strict.decide(sim.examples.getFirst().state()).calibrated());
    }
}
