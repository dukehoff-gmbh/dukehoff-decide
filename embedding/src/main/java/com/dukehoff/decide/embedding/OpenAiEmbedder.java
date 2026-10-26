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

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;

/**
 * {@link Embedder} for OpenAI-compatible {@code POST /embeddings}: OpenAI, llama.cpp ({@code llama-server
 * --embeddings}), vLLM, LM Studio, Ollama. Texts are sent in batches, several requests in flight; timeouts, I/O
 * errors, 429 and 5xx are retried. HTTP/1.1 only.
 *
 * <pre>{@code
 * Embedder embedder = OpenAiEmbedder.builder()
 *         .baseUrl("http://localhost:8081/v1")       // llama-server -hf ggml-org/bge-m3-Q8_0-GGUF --embeddings
 *         .model("bge-m3")
 *         .build();
 * }</pre>
 */
public final class OpenAiEmbedder implements Embedder {

    private static final ObjectMapper JSON = new ObjectMapper();

    private final String model;
    private final String modelId;
    private final URI endpoint;
    private final String apiKey;
    private final int batchSize;
    private final int maxRetries;
    private final Duration timeout;
    private final HttpClient http;
    private final Semaphore permits;

    private OpenAiEmbedder(Builder b) {
        this.model = Objects.requireNonNull(b.model, "model must be set");
        this.modelId = b.modelId != null ? b.modelId : b.model;
        this.endpoint = URI.create(b.baseUrl.replaceAll("/+$", "") + "/embeddings");
        this.apiKey = b.apiKey;
        this.batchSize = b.batchSize;
        this.maxRetries = b.maxRetries;
        this.timeout = b.timeout;
        this.http = b.httpClient != null ? b.httpClient : HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofSeconds(10)).build();
        this.permits = new Semaphore(b.maxConcurrency);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String modelId() {
        return modelId;
    }

    public String model() {
        return model;
    }

    @Override
    public List<float[]> embed(List<String> texts) {
        List<Future<List<float[]>>> batches = new ArrayList<>();
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < texts.size(); i += batchSize) {
                List<String> batch = List.copyOf(texts.subList(i, Math.min(texts.size(), i + batchSize)));
                batches.add(pool.submit(() -> {
                    permits.acquire();
                    try {
                        return send(batch);
                    } finally {
                        permits.release();
                    }
                }));
            }
            List<float[]> out = new ArrayList<>(texts.size());
            for (Future<List<float[]>> f : batches) out.addAll(f.get());
            return out;
        } catch (ExecutionException e) {
            throw e.getCause() instanceof DecideException d ? d : new DecideException(e.getCause().getMessage(), e.getCause());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DecideException("Interrupted while embedding", e);
        }
    }

    /** Embedding is idempotent, so timeouts, I/O errors, 429 and 5xx are retried with exponential backoff. */
    private List<float[]> send(List<String> batch) throws InterruptedException {
        HttpRequest request = request(batch);
        for (int attempt = 0; ; attempt++) {
            HttpResponse<String> response;
            try {
                response = http.send(request, HttpResponse.BodyHandlers.ofString());
            } catch (IOException e) {
                if (attempt >= maxRetries) throw new DecideException("POST " + endpoint + " failed: " + e.getMessage(), e);
                Thread.sleep(500L << attempt);
                continue;
            }
            if ((response.statusCode() == 429 || response.statusCode() >= 500) && attempt < maxRetries) {
                Thread.sleep(500L << attempt);
                continue;
            }
            return parse(response, batch.size());
        }
    }

    private HttpRequest request(List<String> batch) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("input", batch);
        String json;
        try {
            json = JSON.writeValueAsString(body);
        } catch (IOException e) {
            throw new DecideException("Cannot write request", e);
        }
        HttpRequest.Builder rb = HttpRequest.newBuilder(endpoint).timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json));
        if (apiKey != null) rb.header("Authorization", "Bearer " + apiKey);
        return rb.build();
    }

    private List<float[]> parse(HttpResponse<String> response, int expected) {
        if (response.statusCode() != 200) {
            throw new DecideException("POST " + endpoint + " returned " + response.statusCode() + ": " + response.body());
        }
        JsonNode data;
        try {
            data = JSON.readTree(response.body()).path("data");
        } catch (IOException e) {
            throw new DecideException("Cannot parse response of " + endpoint, e);
        }
        float[][] out = new float[expected][];
        for (int j = 0; j < data.size(); j++) {
            JsonNode d = data.get(j);
            int index = d.path("index").asInt(j);
            JsonNode e = d.path("embedding");
            if (index < 0 || index >= expected || !e.isArray()) {
                throw new DecideException("Unexpected embedding response from " + endpoint);
            }
            float[] v = new float[e.size()];
            for (int i = 0; i < v.length; i++) v[i] = (float) e.get(i).asDouble();
            out[index] = v;
        }
        for (float[] v : out) {
            if (v == null) throw new DecideException(endpoint + " returned " + data.size() + " embeddings for " + expected + " texts");
        }
        return List.of(out);
    }

    public static final class Builder {
        private String baseUrl = "https://api.openai.com/v1";
        private String apiKey;
        private String model;
        private String modelId;
        private int batchSize = 32;
        private int maxConcurrency = 4;
        private int maxRetries = 2;
        private Duration timeout = Duration.ofSeconds(60);
        private HttpClient httpClient;

        private Builder() {}

        /** Base URL including {@code /v1}. Default: OpenAI. */
        public Builder baseUrl(String baseUrl) { this.baseUrl = Objects.requireNonNull(baseUrl); return this; }

        /** Sent as {@code Authorization: Bearer ...}; not needed for local servers. */
        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }

        /** The {@code model} field of the request, e.g. {@code bge-m3} or {@code text-embedding-3-small}. */
        public Builder model(String model) { this.model = model; return this; }

        /** Overrides {@link #modelId()} (default: the model name), e.g. to include the quantisation. */
        public Builder modelId(String modelId) { this.modelId = modelId; return this; }

        /** Texts per request. Default 32. */
        public Builder batchSize(int n) {
            if (n < 1) throw new IllegalArgumentException("batch size must be at least 1");
            this.batchSize = n;
            return this;
        }

        /** Requests in flight. Default 4. */
        public Builder maxConcurrency(int n) {
            if (n < 1) throw new IllegalArgumentException("max concurrency must be at least 1");
            this.maxConcurrency = n;
            return this;
        }

        /** Retries after a timeout, I/O error, 429 or 5xx, waiting 0.5 s, 1 s, 2 s, ... Default 2. */
        public Builder maxRetries(int n) {
            if (n < 0) throw new IllegalArgumentException("max retries must not be negative");
            this.maxRetries = n;
            return this;
        }

        /** Per request. Default 60 s. */
        public Builder timeout(Duration timeout) { this.timeout = Objects.requireNonNull(timeout); return this; }

        public Builder httpClient(HttpClient httpClient) { this.httpClient = httpClient; return this; }

        public OpenAiEmbedder build() {
            return new OpenAiEmbedder(this);
        }
    }
}
