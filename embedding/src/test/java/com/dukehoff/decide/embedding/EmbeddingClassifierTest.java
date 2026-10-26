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
package com.dukehoff.decide.embedding;

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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingClassifierTest {

    enum Team { BILLING, SHIPPING, ACCOUNT }

    @Question("Route a customer message")
    record Route(@Question("Which team?") Team team, @Question("Is the customer angry?") boolean angry) {}

    /** Bag of words hashed into a few dimensions: enough structure for a linear probe. */
    static final class HashEmbedder implements Embedder {
        final String id;
        final int dim;
        final AtomicInteger calls = new AtomicInteger();

        HashEmbedder(String id, int dim) {
            this.id = id;
            this.dim = dim;
        }

        public String modelId() {
            return id;
        }

        public List<float[]> embed(List<String> texts) {
            calls.incrementAndGet();
            return texts.stream().map(t -> {
                float[] v = new float[dim];
                for (String w : t.toLowerCase(Locale.ROOT).split("\\W+")) v[Math.floorMod(w.hashCode(), dim)] += 1;
                return v;
            }).toList();
        }
    }

    private static final String[] TEAM_WORDS = {"invoice refund payment", "parcel delivery courier", "password login profile"};

    static List<Labeled<Route>> messages(int n, long seed) {
        Random random = new Random(seed);
        List<Labeled<Route>> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Team team = Team.values()[random.nextInt(3)];
            boolean angry = random.nextBoolean();
            String[] w = TEAM_WORDS[team.ordinal()].split(" ");
            String text = "about my " + w[random.nextInt(3)] + " " + w[random.nextInt(3)] + " "
                    + (angry ? "outrageous furious" : "thanks regards") + " order " + random.nextInt(1000);
            out.add(new Labeled<>(State.of(text), new Route(team, angry)));
        }
        return out;
    }

    @Test
    void learnsAndCalibratesThroughTheDecider() {
        HashEmbedder embedder = new HashEmbedder("hash-64", 64);
        EmbeddingClassifier model = EmbeddingClassifier.train(embedder, Route.class, messages(300, 1));
        DecisionTask<Route> task = Decider.builder().backend(model).build().task(Route.class);

        List<Labeled<Route>> test = messages(100, 2);
        long correct = test.stream().filter(e -> task.decide(e.state()).value().equals(e.expected())).count();
        assertTrue(correct >= 90, "correct: " + correct);

        Calibration.CalibrationReport report = Calibration.fit(task, messages(100, 3), messages(100, 4));
        assertEquals(model.modelId(), report.profile().modelId());
        assertTrue(model.modelId().startsWith("embedding-"), model.modelId());
    }

    @Test
    void scoreAllEmbedsInOneCall() {
        HashEmbedder embedder = new HashEmbedder("hash-64", 64);
        EmbeddingClassifier model = EmbeddingClassifier.train(embedder, Route.class, messages(50, 1));
        int before = embedder.calls.get();
        TaskSchema schema = TaskSchema.of(Route.class);
        model.scoreAll(messages(20, 2).stream().map(e -> new ScoringRequest(e.state(), schema)).toList());
        assertEquals(before + 1, embedder.calls.get());
    }

    @Test
    void savedModelScoresTheSameAndChecksTheEmbedder(@TempDir Path dir) {
        HashEmbedder embedder = new HashEmbedder("hash-64", 64);
        EmbeddingClassifier model = EmbeddingClassifier.train(embedder, Route.class, messages(100, 1));
        Path file = dir.resolve("route.model");
        model.save(file);
        EmbeddingClassifier loaded = EmbeddingClassifier.load(file, embedder);

        assertEquals(model.modelId(), loaded.modelId());
        assertEquals("hash-64", loaded.embedderId());
        ScoringRequest request = new ScoringRequest(State.of("refund outrageous"), TaskSchema.of(Route.class));
        for (String field : List.of("team", "angry")) {
            assertArrayEquals(model.score(request).get(field).logProbs(), loaded.score(request).get(field).logProbs());
        }

        DecideException other = assertThrows(DecideException.class,
                () -> EmbeddingClassifier.load(file, new HashEmbedder("other-model", 64)));
        assertTrue(other.getMessage().contains("hash-64"), other.getMessage());
    }

    @Test
    void modelIdFollowsWeightsAndEmbedder() {
        String a = EmbeddingClassifier.train(new HashEmbedder("a", 32), Route.class, messages(50, 1)).modelId();
        assertEquals(a, EmbeddingClassifier.train(new HashEmbedder("a", 32), Route.class, messages(50, 1)).modelId());
        assertNotEquals(a, EmbeddingClassifier.train(new HashEmbedder("b", 32), Route.class, messages(50, 1)).modelId());
        assertEquals("v1", EmbeddingClassifier.trainer().modelId("v1")
                .train(new HashEmbedder("a", 32), Route.class, messages(50, 1)).modelId());
    }

    @Test
    void refusesVectorsOfAnotherLength(@TempDir Path dir) {
        EmbeddingClassifier model = EmbeddingClassifier.train(new HashEmbedder("same", 32), Route.class, messages(50, 1));
        model.save(dir.resolve("m"));
        EmbeddingClassifier wrongDim = EmbeddingClassifier.load(dir.resolve("m"), new HashEmbedder("same", 16));
        DecideException e = assertThrows(DecideException.class,
                () -> wrongDim.score(new ScoringRequest(State.of("refund"), TaskSchema.of(Route.class))));
        assertTrue(e.getMessage().contains("32-dimensional"), e.getMessage());
    }
}
