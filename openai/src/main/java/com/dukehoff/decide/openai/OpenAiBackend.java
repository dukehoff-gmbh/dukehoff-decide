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
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Scores fields through an OpenAI-compatible {@code /chat/completions} endpoint with
 * {@code logprobs} and {@code top_logprobs}: OpenAI, llama.cpp server, vLLM, LM Studio, Ollama.
 *
 * <p>One request per field. Each asks for a single token with temperature 0, and the
 * log-probabilities of the option labels among the top-k candidates become the raw scores
 * (see {@link LabelPrompt}). Labels outside the top-k get the lowest log-probability the server
 * returned, which is an upper bound for them. Requests run concurrently, up to
 * {@link Builder#maxConcurrency(int)} at a time; failed attempts are retried (see {@link Builder#maxRetries(int)}).
 *
 * <pre>
 * ScoringBackend backend = OpenAiBackend.builder()
 *     .baseUrl("http://localhost:8080/v1")   // llama.cpp server
 *     .model("qwen2.5-3b-instruct")
 *     .build();
 * </pre>
 */
public final class OpenAiBackend implements ScoringBackend {

    private static final ObjectMapper JSON = new ObjectMapper();

    private final String baseUrl;
    private final URI endpoint;
    private final String apiKey;
    private final String model;
    private final String modelId;
    private final int topLogprobs;
    private final Duration timeout;
    private final int maxRetries;
    private final Map<String, Object> params;
    private final HttpClient http;
    private final Semaphore permits;

    private OpenAiBackend(Builder b) {
        String base = b.baseUrl.endsWith("/") ? b.baseUrl.substring(0, b.baseUrl.length() - 1) : b.baseUrl;
        this.baseUrl = base;
        this.endpoint = URI.create(base + "/chat/completions");
        this.apiKey = b.apiKey;
        this.model = Objects.requireNonNull(b.model, "model");
        this.modelId = b.modelId != null ? b.modelId : b.model;
        this.topLogprobs = b.topLogprobs;
        this.timeout = b.timeout;
        this.maxRetries = b.maxRetries;
        this.params = Map.copyOf(b.params);
        this.http = b.httpClient != null ? b.httpClient
                : HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
                        .connectTimeout(Duration.ofSeconds(10)).build();
        this.permits = new Semaphore(b.maxConcurrency);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String modelId() {
        return modelId;
    }

    /** Base URL without trailing slash, e.g. {@code http://localhost:8080/v1}. */
    public String baseUrl() {
        return baseUrl;
    }

    /** The {@code model} field sent with each request. */
    public String model() {
        return model;
    }

    /**
     * Lists the model ids the server offers ({@code GET /models}); a cheap reachability check.
     * Local servers often report a different id (e.g. the file name) than the alias used in requests.
     *
     * @throws DecideException if the server cannot be reached or does not answer with 2xx
     */
    public List<String> models(Duration timeout) {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(baseUrl + "/models")).timeout(timeout).GET();
        if (apiKey != null && !apiKey.isBlank()) b.header("Authorization", "Bearer " + apiKey);
        HttpResponse<String> response;
        try {
            response = http.send(b.build(), HttpResponse.BodyHandlers.ofString());
        } catch (IOException e) {
            throw new DecideException("GET " + baseUrl + "/models failed: " + e, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DecideException("Interrupted while calling " + baseUrl + "/models", e);
        }
        if (response.statusCode() / 100 != 2) {
            throw new DecideException("GET " + baseUrl + "/models returned " + response.statusCode() + ": "
                    + abbreviate(response.body()));
        }
        List<String> ids = new ArrayList<>();
        try {
            for (JsonNode m : JSON.readTree(response.body()).path("data")) {
                if (m.path("id").isTextual()) ids.add(m.path("id").asText());
            }
        } catch (JsonProcessingException e) {
            throw new DecideException("GET " + baseUrl + "/models returned invalid JSON: " + abbreviate(response.body()), e);
        }
        return ids;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        return scoreAll(List.of(request)).getFirst();
    }

    @Override
    public List<Map<String, RawScores>> scoreAll(List<ScoringRequest> requests) {
        List<Map<String, CompletableFuture<RawScores>>> pending = new ArrayList<>();
        for (ScoringRequest r : requests) {
            Map<String, CompletableFuture<RawScores>> fields = new LinkedHashMap<>();
            for (FieldSpec f : r.schema().fields()) {
                fields.put(f.name(), score(LabelPrompt.of(r.schema(), f, r.state())));
            }
            pending.add(fields);
        }
        List<Map<String, RawScores>> out = new ArrayList<>();
        for (Map<String, CompletableFuture<RawScores>> fields : pending) {
            Map<String, RawScores> scores = new LinkedHashMap<>();
            fields.forEach((name, f) -> scores.put(name, join(f)));
            out.add(scores);
        }
        return out;
    }

    private CompletableFuture<RawScores> score(LabelPrompt prompt) {
        HttpRequest request = request(prompt);
        try {
            permits.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DecideException("Interrupted while waiting for a free connection", e);
        }
        return send(request, 0)
                .whenComplete((r, e) -> permits.release())
                .thenApply(response -> parse(prompt, response));
    }

    /** Scoring is idempotent, so timeouts, I/O errors, 429 and 5xx are retried with exponential backoff. */
    private CompletableFuture<HttpResponse<String>> send(HttpRequest request, int attempt) {
        return http.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .handle((response, error) -> {
                    if (attempt < maxRetries && retryable(response, error)) {
                        Executor later = CompletableFuture.delayedExecutor(500L << attempt, TimeUnit.MILLISECONDS);
                        return CompletableFuture.supplyAsync(() -> null, later)
                                .thenCompose(x -> send(request, attempt + 1));
                    }
                    return error != null ? CompletableFuture.<HttpResponse<String>>failedFuture(error)
                            : CompletableFuture.completedFuture(response);
                })
                .thenCompose(f -> f);
    }

    private static boolean retryable(HttpResponse<String> response, Throwable error) {
        if (error != null) {
            Throwable cause = error instanceof CompletionException && error.getCause() != null ? error.getCause() : error;
            return cause instanceof IOException;
        }
        return response.statusCode() == 429 || response.statusCode() >= 500;
    }

    HttpRequest request(LabelPrompt prompt) {
        ObjectNode body = JSON.createObjectNode();
        body.put("model", model);
        var messages = body.putArray("messages");
        messages.addObject().put("role", "system").put("content", prompt.system());
        messages.addObject().put("role", "user").put("content", prompt.user());
        body.put("max_tokens", 1);
        body.put("temperature", 0);
        body.put("logprobs", true);
        body.put("top_logprobs", topLogprobs);
        params.forEach((k, v) -> body.set(k, JSON.valueToTree(v)));

        HttpRequest.Builder b = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()));
        if (apiKey != null && !apiKey.isBlank()) b.header("Authorization", "Bearer " + apiKey);
        return b.build();
    }

    private RawScores parse(LabelPrompt prompt, HttpResponse<String> response) {
        if (response.statusCode() / 100 != 2) {
            throw new DecideException("POST " + endpoint + " returned " + response.statusCode() + ": "
                    + abbreviate(response.body()));
        }
        JsonNode first;
        try {
            first = JSON.readTree(response.body()).path("choices").path(0).path("logprobs").path("content").path(0);
        } catch (JsonProcessingException e) {
            throw new DecideException("POST " + endpoint + " returned invalid JSON: " + abbreviate(response.body()), e);
        }
        if (first.isMissingNode()) {
            throw new DecideException("POST " + endpoint + " returned no logprobs for model '" + model
                    + "'. Does the server support \"logprobs\": true? Response: " + abbreviate(response.body()));
        }
        Map<String, Double> candidates = new LinkedHashMap<>();
        double lowest = Double.POSITIVE_INFINITY;
        for (JsonNode t : first.path("top_logprobs")) {
            lowest = Math.min(lowest, add(candidates, t));
        }
        // the sampled token, in case the server ignores top_logprobs
        lowest = Math.min(lowest, add(candidates, first));
        double missing = lowest == Double.POSITIVE_INFINITY ? Double.NEGATIVE_INFINITY : lowest;
        return new RawScores(prompt.scores(candidates, missing));
    }

    /** @return the token's log-probability, or +Infinity if the node has none */
    private static double add(Map<String, Double> candidates, JsonNode t) {
        if (!t.path("token").isTextual() || !t.path("logprob").isNumber()) return Double.POSITIVE_INFINITY;
        double lp = t.path("logprob").asDouble();
        candidates.putIfAbsent(t.path("token").asText(), lp);
        return lp;
    }

    private static RawScores join(CompletableFuture<RawScores> f) {
        try {
            return f.join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof DecideException j) throw j;
            throw new DecideException("Scoring request failed: " + e.getCause(), e.getCause());
        }
    }

    private static String abbreviate(String s) {
        return s.length() <= 500 ? s : s.substring(0, 500) + "...";
    }

    public static final class Builder {
        private String baseUrl = "https://api.openai.com/v1";
        private String apiKey;
        private String model;
        private String modelId;
        private int topLogprobs = 20;
        private int maxConcurrency = 8;
        private int maxRetries = 2;
        private Duration timeout = Duration.ofSeconds(60);
        private final Map<String, Object> params = new LinkedHashMap<>();
        private HttpClient httpClient;

        private Builder() {}

        /** Base URL up to and including {@code /v1}. Default: OpenAI. */
        public Builder baseUrl(String baseUrl) { this.baseUrl = Objects.requireNonNull(baseUrl); return this; }

        /** Sent as bearer token. Optional for local servers. */
        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }

        /** The {@code model} field of the request. Required. */
        public Builder model(String model) { this.model = model; return this; }

        /**
         * Id for calibration profiles; defaults to {@link #model(String)}. Set it to something more
         * specific (quantisation, server, prompt version) when those differ between deployments.
         */
        public Builder modelId(String modelId) { this.modelId = modelId; return this; }

        /** Candidates per request. Default 20, the maximum OpenAI accepts. */
        public Builder topLogprobs(int n) {
            if (n < 1) throw new IllegalArgumentException("topLogprobs must be >= 1: " + n);
            this.topLogprobs = n;
            return this;
        }

        /** Requests in flight at once. Default 8. */
        public Builder maxConcurrency(int n) {
            if (n < 1) throw new IllegalArgumentException("maxConcurrency must be >= 1: " + n);
            this.maxConcurrency = n;
            return this;
        }

        /** Retries after a timeout, I/O error, 429 or 5xx, waiting 0.5 s, 1 s, 2 s, ... Default 2. */
        public Builder maxRetries(int n) {
            if (n < 0) throw new IllegalArgumentException("maxRetries must be >= 0: " + n);
            this.maxRetries = n;
            return this;
        }

        /** Per attempt. Default 60 seconds. */
        public Builder timeout(Duration timeout) { this.timeout = Objects.requireNonNull(timeout); return this; }

        /** Extra request body field, e.g. {@code seed}, or overriding {@code max_tokens}. */
        public Builder param(String name, Object value) { params.put(name, value); return this; }

        public Builder httpClient(HttpClient httpClient) { this.httpClient = httpClient; return this; }

        public OpenAiBackend build() {
            return new OpenAiBackend(this);
        }
    }
}
