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
package com.dukehoff.decide.openai;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/** Runs against a local JDK HTTP server that plays back canned responses. */
class OpenAiBackendTest {

    @Question("How urgent is this ticket?")
    enum Priority {
        @Option("Production is down or data is being lost") BLOCKER,
        @Option("A feature is broken for many users, no workaround") HIGH,
        @Option("Something is broken but there is a workaround") NORMAL,
        @Option("Question, cosmetic issue or feature wish") LOW
    }

    record Ticket(@Question("How urgent is this ticket?") Priority priority,
                  @Question("Does the customer threaten to cancel?") boolean churnRisk) {}

    static final ObjectMapper JSON = new ObjectMapper();

    HttpServer server;
    final List<JsonNode> requests = new CopyOnWriteArrayList<>();
    final List<String> authHeaders = new CopyOnWriteArrayList<>();
    volatile Function<JsonNode, Reply> handler;

    record Reply(int status, String body) {}

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.createContext("/v1/chat/completions", exchange -> {
            JsonNode body = JSON.readTree(exchange.getRequestBody());
            requests.add(body);
            String auth = exchange.getRequestHeaders().getFirst("Authorization");
            if (auth != null) authHeaders.add(auth);
            Reply reply = handler.apply(body);
            byte[] bytes = reply.body().getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(reply.status(), bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    OpenAiBackend.Builder backend() {
        return OpenAiBackend.builder()
                .baseUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/v1/")
                .model("qwen2.5-3b-instruct");
    }

    static String resource(String name) throws IOException {
        try (InputStream in = OpenAiBackendTest.class.getResourceAsStream("/" + name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /** A minimal chat completion whose single generated token has the given top candidates. */
    static String completion(Map<String, Double> top) {
        var content = JSON.createObjectNode();
        var candidates = content.putArray("top_logprobs");
        top.forEach((token, lp) -> candidates.addObject().put("token", token).put("logprob", lp));
        var root = JSON.createObjectNode();
        root.putArray("choices").addObject().putObject("logprobs").putArray("content").add(content);
        return root.toString();
    }

    @Test
    void sendsOneSingleTokenRequestPerFieldWithLogprobs() throws IOException {
        String priority = resource("priority-response.json");
        handler = body -> new Reply(200, priority);

        backend().apiKey("sk-test").param("seed", 42).build()
                .score(new ScoringRequest(State.of("Server down!"),
                        Decider.builder().backend(backend().build()).build().task(Ticket.class).schema()));

        assertEquals(2, requests.size());
        JsonNode r = requests.stream()
                .filter(b -> b.at("/messages/1/content").asText().contains("How urgent")).findFirst().orElseThrow();
        assertEquals("qwen2.5-3b-instruct", r.get("model").asText());
        assertEquals(1, r.get("max_tokens").asInt());
        assertEquals(0, r.get("temperature").asInt());
        assertTrue(r.get("logprobs").asBoolean());
        assertEquals(20, r.get("top_logprobs").asInt());
        assertEquals(42, r.get("seed").asInt());
        assertEquals("system", r.at("/messages/0/role").asText());
        assertTrue(r.at("/messages/1/content").asText().contains("A) BLOCKER: Production is down"));
        assertEquals(List.of("Bearer sk-test", "Bearer sk-test"), authHeaders);
    }

    @Test
    void mapsTopLogprobsToOptionOrder() throws IOException {
        String priority = resource("priority-response.json");
        handler = body -> new Reply(200, priority);
        var schema = Decider.builder().backend(backend().build()).build().task(Priority.class).schema();

        Map<String, RawScores> scores = backend().build()
                .score(new ScoringRequest(State.of("Server down!"), schema));

        double[] lp = scores.get(schema.fields().getFirst().name()).logProbs();
        double a = Math.log(Math.exp(-0.1625) + Math.exp(-3.9)); // "A" and " A" merged
        // D is not among the candidates and gets the lowest returned log-prob
        assertArrayEquals(new double[] {a, -2.05, -7.5, -7.5}, lp, 1e-9);
    }

    /** Recorded from llama.cpp server b11206 with Qwen2.5-1.5B-Instruct Q4_K_M. */
    @Test
    void parsesRecordedLlamaCppResponse() throws IOException {
        String recorded = resource("llamacpp-priority-response.json");
        handler = body -> new Reply(200, recorded);
        var schema = Decider.builder().backend(backend().build()).build().task(Priority.class).schema();

        double[] lp = backend().build().score(new ScoringRequest(State.of("Server down!"), schema))
                .get(Decision.VALUE).logProbs();

        // "A", " A" and "(A" are merged; "AB", "AA", "a" and "BLOCK" are not labels
        double a = Math.log(Math.exp(-9.108004451263696e-05) + Math.exp(-19.770) + Math.exp(-20.840));
        assertArrayEquals(new double[] {a, -9.3816, -12.1055, -13.6545}, lp, 1e-3);
    }

    /** Recorded from Ollama 0.34.4 (/v1/chat/completions) with qwen2.5:1.5b (Q4_K_M). */
    @Test
    void parsesRecordedOllamaResponse() throws IOException {
        String recorded = resource("ollama-priority-response.json");
        handler = body -> new Reply(200, recorded);
        var schema = Decider.builder().backend(backend().build()).build().task(Priority.class).schema();

        double[] lp = backend().build().score(new ScoringRequest(State.of("Server down!"), schema))
                .get(Decision.VALUE).logProbs();

        // "A", " A", "(A" and "'A" are merged; the Cyrillic look-alike "А" is not
        double a = Math.log(Math.exp(-0.00025818104040808976) + Math.exp(-19.93551254272461)
                + Math.exp(-20.42678451538086) + Math.exp(-21.13203239440918));
        assertArrayEquals(new double[] {a, -8.295401573181152, -11.742058753967285, -14.620783805847168}, lp, 1e-12);
    }

    @Test
    void decidesATypedRecordEndToEnd() {
        handler = body -> {
            String prompt = body.at("/messages/1/content").asText();
            return new Reply(200, prompt.contains("How urgent")
                    ? completion(Map.of("A", Math.log(0.8), "B", Math.log(0.15), "C", Math.log(0.05)))
                    : completion(Map.of("B", Math.log(0.7), "A", Math.log(0.3))));
        };
        Decider decider = Decider.builder().backend(backend().build()).build();

        List<Decision<Ticket>> ds = decider.task(Ticket.class)
                .decideAll(List.of(State.of("Server down!"), State.of("Server down again!")));

        assertEquals(2, ds.size());
        assertEquals(4, requests.size());
        for (Decision<Ticket> d : ds) {
            assertEquals(new Ticket(Priority.BLOCKER, false), d.value());
            assertEquals(0.7, d.confidence(), 1e-6);
        }
    }

    @Test
    void noLabelAmongCandidatesMeansUniform() {
        handler = body -> new Reply(200, completion(Map.of("I", -0.5, "Sure", -1.0)));
        Decision<Priority> d = Decider.builder().backend(backend().build()).build()
                .task(Priority.class).decide(State.of("hello"));
        assertEquals(0.25, d.confidence(), 1e-9);
    }

    @Test
    void reportsHttpErrors() {
        handler = body -> new Reply(401, "{\"error\":{\"message\":\"Incorrect API key provided\"}}");
        var task = Decider.builder().backend(backend().build()).build().task(Priority.class);
        var e = assertThrows(DecideException.class, () -> task.decide(State.of("x")));
        assertTrue(e.getMessage().contains("401"), e.getMessage());
        assertTrue(e.getMessage().contains("Incorrect API key"), e.getMessage());
    }

    @Test
    void retriesServerErrorsAndTimeouts() {
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        handler = body -> switch (calls.incrementAndGet()) {
            case 1 -> new Reply(503, "{\"error\":\"loading model\"}");
            case 2 -> {
                try {
                    Thread.sleep(1_000); // longer than the client timeout
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                yield new Reply(200, completion(Map.of("B", -0.1)));
            }
            default -> new Reply(200, completion(Map.of("A", -0.1)));
        };
        var task = Decider.builder().backend(backend().timeout(java.time.Duration.ofMillis(300)).build()).build()
                .task(Priority.class);

        assertEquals(Priority.BLOCKER, task.decide(State.of("x")).value());
        assertEquals(3, calls.get());
    }

    @Test
    void givesUpAfterMaxRetries() {
        handler = body -> new Reply(500, "{\"error\":\"boom\"}");
        var task = Decider.builder().backend(backend().maxRetries(1).build()).build().task(Priority.class);
        var e = assertThrows(DecideException.class, () -> task.decide(State.of("x")));
        assertTrue(e.getMessage().contains("500"), e.getMessage());
        assertEquals(2, requests.size());
    }

    @Test
    void explainsMissingLogprobs() {
        handler = body -> new Reply(200, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"A\"}}]}");
        var task = Decider.builder().backend(backend().build()).build().task(Priority.class);
        var e = assertThrows(DecideException.class, () -> task.decide(State.of("x")));
        assertTrue(e.getMessage().contains("no logprobs"), e.getMessage());
    }

    @Test
    void listsServerModels() {
        server.createContext("/v1/models", exchange -> {
            byte[] bytes = "{\"object\":\"list\",\"data\":[{\"id\":\"Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M\"}]}"
                    .getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        OpenAiBackend backend = backend().build();
        assertEquals(List.of("Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"), backend.models(java.time.Duration.ofSeconds(2)));
        assertEquals("http://127.0.0.1:" + server.getAddress().getPort() + "/v1", backend.baseUrl());
        assertEquals("qwen2.5-3b-instruct", backend.model());

        OpenAiBackend unreachable = OpenAiBackend.builder().baseUrl("http://127.0.0.1:1/v1").model("x").build();
        assertThrows(DecideException.class, () -> unreachable.models(java.time.Duration.ofSeconds(2)));
    }

    @Test
    void modelIdDefaultsToModelAndCanBeOverridden() {
        assertEquals("qwen2.5-3b-instruct", backend().build().modelId());
        assertEquals("qwen2.5-3b-instruct-q4_k_m", backend().modelId("qwen2.5-3b-instruct-q4_k_m").build().modelId());
    }
}
