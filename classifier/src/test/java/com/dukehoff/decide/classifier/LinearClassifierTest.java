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
package com.dukehoff.decide.classifier;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.calibration.Calibration.Labeled;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.spi.TaskSchema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LinearClassifierTest {

    enum Team { BILLING, SHIPPING, ACCOUNT }

    enum Department { BILLING, SHIPPING, ACCOUNT, LEGAL }

    @Question("Route a customer message")
    record Route(@Question("Which team?") Team team, @Question("Is the customer angry?") boolean angry) {}

    @Question("Route a customer message")
    record Other(@Question("Which department?") Department team) {}

    private static final String[][] WORDS = {
            {"invoice", "refund", "charged twice", "payment", "bill"},
            {"parcel", "delivery", "tracking number", "courier", "package"},
            {"password", "login", "profile", "username", "two-factor"}};
    private static final String[] ANGRY = {"this is outrageous", "worst service ever", "I am furious"};
    private static final String[] CALM = {"thanks in advance", "have a nice day", "kind regards"};

    /** Messages whose team and mood are named by a few words, plus filler. */
    static List<Labeled<Route>> messages(int n, long seed) {
        Random random = new Random(seed);
        List<Labeled<Route>> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Team team = Team.values()[random.nextInt(3)];
            boolean angry = random.nextBoolean();
            String[] w = WORDS[team.ordinal()];
            String text = "Hello, about my " + w[random.nextInt(w.length)] + " and the " + w[random.nextInt(w.length)]
                    + ": " + (angry ? ANGRY : CALM)[random.nextInt(3)] + ". Order " + random.nextInt(100000);
            out.add(new Labeled<>(State.of(text), new Route(team, angry)));
        }
        return out;
    }

    @Test
    void learnsAndDecidesThroughTheDecider() {
        LinearClassifier model = LinearClassifier.train(Route.class, messages(400, 1));
        DecisionTask<Route> task = Decider.builder().backend(model).build().task(Route.class);

        List<Labeled<Route>> test = messages(100, 2);
        long correct = test.stream().filter(e -> task.decide(e.state()).value().equals(e.expected())).count();
        assertTrue(correct >= 95, "correct: " + correct);
        assertEquals(Team.SHIPPING, task.decide(State.of("Where is my parcel? The tracking number does not work"))
                .value().team());
    }

    @Test
    void calibratesLikeAnyOtherBackend() {
        LinearClassifier model = LinearClassifier.train(Route.class, messages(400, 1));
        DecisionTask<Route> task = Decider.builder().backend(model).build().task(Route.class);

        Calibration.CalibrationReport report = Calibration.fit(task, messages(150, 3), messages(150, 4));
        assertEquals(model.modelId(), report.profile().modelId());
        assertTrue(report.fields().stream().allMatch(f -> f.accuracy() > 0.9), report.toString());
    }

    @Test
    void savedModelScoresTheSame(@TempDir Path dir) throws IOException {
        LinearClassifier model = LinearClassifier.trainer().hashBits(14).train(Route.class, messages(200, 1));
        Path file = dir.resolve("route.model");
        model.save(file);
        LinearClassifier loaded = LinearClassifier.load(file);

        assertEquals(model.modelId(), loaded.modelId());
        assertEquals(model.taskId(), loaded.taskId());
        ScoringRequest request = new ScoringRequest(State.of("I was charged twice, this is outrageous"),
                TaskSchema.of(Route.class));
        for (String field : List.of("team", "angry")) {
            assertArrayEquals(model.score(request).get(field).logProbs(), loaded.score(request).get(field).logProbs());
        }
        assertTrue(Files.size(file) < 200_000, "only non-zero weights are stored: " + Files.size(file));

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        loaded.save(bytes);
        assertEquals(model.modelId(), LinearClassifier.load(new ByteArrayInputStream(bytes.toByteArray())).modelId());
    }

    @Test
    void modelIdFollowsTheWeights() {
        LinearClassifier.Trainer trainer = LinearClassifier.trainer().hashBits(12);
        String id = trainer.train(Route.class, messages(100, 1)).modelId();
        assertTrue(id.startsWith("linear-"), id);
        assertEquals(id, trainer.train(Route.class, messages(100, 1)).modelId());
        assertNotEquals(id, trainer.train(Route.class, messages(101, 1)).modelId());
        assertNotEquals(id, trainer.seed(7).train(Route.class, messages(100, 1)).modelId());
        assertEquals("route-v1", trainer.modelId("route-v1").train(Route.class, messages(100, 1)).modelId());
    }

    @Test
    void refusesOtherTasksAndChangedOptions() {
        LinearClassifier model = LinearClassifier.trainer().hashBits(12).train(Route.class, messages(50, 1));
        State state = State.of("refund please");

        DecideException other = assertThrows(DecideException.class,
                () -> model.score(new ScoringRequest(state, TaskSchema.of(Other.class))));
        assertTrue(other.getMessage().contains("was trained for"), other.getMessage());

        TaskSchema route = TaskSchema.of(Route.class);
        TaskSchema changed = new TaskSchema(route.taskId(), route.type(), route.kind(), route.instructions(),
                TaskSchema.of(Other.class).fields());
        DecideException retrain = assertThrows(DecideException.class,
                () -> model.score(new ScoringRequest(state, changed)));
        assertTrue(retrain.getMessage().contains("retrain"), retrain.getMessage());
    }

    @Test
    void trainsBooleanTasks() {
        List<Labeled<Boolean>> spam = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            spam.add(new Labeled<>(State.of("cheap pills, win money now " + i), true));
            spam.add(new Labeled<>(State.of("meeting notes for tuesday " + i), false));
        }
        LinearClassifier model = LinearClassifier.trainer().hashBits(12).train(Boolean.class, "Is it spam?", spam);
        DecisionTask<Boolean> task = Decider.builder().backend(model).build().task(Boolean.class, "Is it spam?");

        assertEquals(true, task.decide(State.of("win money with cheap pills")).value());
        assertEquals(false, task.decide(State.of("notes from the meeting")).value());
    }

    @Test
    void rejectsBadSettings() {
        assertThrows(IllegalArgumentException.class, () -> LinearClassifier.trainer().hashBits(30));
        assertThrows(IllegalArgumentException.class, () -> LinearClassifier.trainer().epochs(0));
        assertThrows(IllegalArgumentException.class, () -> LinearClassifier.train(Route.class, List.of()));
        assertThrows(IOException.class, () -> LinearClassifier.load(new ByteArrayInputStream(new byte[] {1, 2, 3})));
    }
}
