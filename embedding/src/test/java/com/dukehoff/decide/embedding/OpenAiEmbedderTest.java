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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Runs against a local JDK HTTP server that answers like {@code /v1/embeddings}. */
class OpenAiEmbedderTest {

    static final ObjectMapper JSON = new ObjectMapper();

    HttpServer server;
    final List<JsonNode> requests = new CopyOnWriteArrayList<>();
    final List<String> authHeaders = new CopyOnWriteArrayList<>();
    volatile Function<JsonNode, Reply> handler = OpenAiEmbedderTest::lengths;

    record Reply(int status, String body) {}

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/embeddings", exchange -> {
            try (InputStream in = exchange.getRequestBody()) {
                JsonNode request = JSON.readTree(new String(in.readAllBytes(), StandardCharsets.UTF_8));
                requests.add(request);
                String auth = exchange.getRequestHeaders().getFirst("Authorization");
                if (auth != null) authHeaders.add(auth);
                Reply reply = handler.apply(request);
                byte[] body = reply.body().getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(reply.status(), body.length);
                exchange.getResponseBody().write(body);
            } finally {
                exchange.close();
            }
        });
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    /** Each text becomes [length, index in batch]; the data array is returned in reverse order. */
    static Reply lengths(JsonNode request) {
        ObjectNode root = JSON.createObjectNode();
        ArrayNode data = root.putArray("data");
        JsonNode input = request.get("input");
        for (int i = input.size() - 1; i >= 0; i--) {
            ObjectNode d = data.addObject();
            d.put("index", i);
            d.putArray("embedding").add(input.get(i).asText().length()).add(i);
        }
        return new Reply(200, root.toString());
    }

    OpenAiEmbedder.Builder embedder() {
        return OpenAiEmbedder.builder().baseUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/v1/")
                .model("bge-m3").maxRetries(1).timeout(Duration.ofSeconds(5));
    }

    @Test
    void batchesAndKeepsTheOrder() {
        List<float[]> out = embedder().batchSize(2).apiKey("secret").build()
                .embed(List.of("a", "bb", "ccc", "dddd", "eeeee"));

        assertEquals(3, requests.size());
        assertTrue(requests.stream().allMatch(r -> r.get("model").asText().equals("bge-m3")));
        for (int i = 0; i < 5; i++) {
            assertEquals(i + 1, out.get(i)[0]);
            assertEquals(i % 2, out.get(i)[1]);
        }
        assertEquals(List.of("Bearer secret", "Bearer secret", "Bearer secret"), authHeaders);
    }

    @Test
    void retriesServerErrors() {
        AtomicInteger calls = new AtomicInteger();
        handler = r -> calls.incrementAndGet() == 1 ? new Reply(503, "loading model") : lengths(r);

        assertEquals(4, embedder().build().embed("four")[0]);
        assertEquals(2, calls.get());
    }

    @Test
    void reportsClientErrorsAndShortResponses() {
        handler = r -> new Reply(400, "{\"error\":\"input too long\"}");
        DecideException bad = assertThrows(DecideException.class, () -> embedder().build().embed("x"));
        assertTrue(bad.getMessage().contains("400") && bad.getMessage().contains("input too long"), bad.getMessage());

        handler = r -> new Reply(200, "{\"data\":[]}");
        DecideException missing = assertThrows(DecideException.class, () -> embedder().build().embed("x"));
        assertTrue(missing.getMessage().contains("0 embeddings for 1"), missing.getMessage());
    }

    @Test
    void modelIdDefaultsToTheModel() {
        assertEquals("bge-m3", embedder().build().modelId());
        assertEquals("bge-m3-q8", embedder().modelId("bge-m3-q8").build().modelId());
    }
}
