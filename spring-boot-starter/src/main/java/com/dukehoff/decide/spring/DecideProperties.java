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
package com.dukehoff.decide.spring;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

import java.time.Duration;

/**
 * Configuration under {@code dukehoff.decide.*}.
 *
 * <pre>
 * dukehoff:
 *   decide:
 *     openai:
 *       base-url: http://localhost:8080/v1
 *       model: qwen2.5-7b-instruct-q4_k_m
 *     calibration:
 *       location: classpath:decide/calibration
 *       required: true
 * </pre>
 *
 * @param openai      the OpenAI-compatible backend; created only if dukehoff.decide.openai.model is set
 * @param calibration where calibration profiles are loaded from
 * @param metrics     Micrometer metrics per task
 */
@ConfigurationProperties("dukehoff.decide")
public record DecideProperties(@DefaultValue OpenAi openai, @DefaultValue Calibration calibration,
                               @DefaultValue Metrics metrics) {

    /**
     * @param baseUrl        up to and including /v1; defaults to OpenAI
     * @param apiKey         sent as bearer token; optional for local servers
     * @param model          the model field of each request
     * @param modelId        key for calibration profiles; defaults to model
     * @param topLogprobs    candidate tokens per request
     * @param maxConcurrency requests in flight at once
     * @param maxRetries     retries after timeouts, I/O errors, 429 and 5xx
     * @param timeout        per attempt
     */
    public record OpenAi(String baseUrl, String apiKey, String model, String modelId,
                         @DefaultValue("20") int topLogprobs,
                         @DefaultValue("8") int maxConcurrency,
                         @DefaultValue("2") int maxRetries,
                         @DefaultValue("60s") Duration timeout) {}

    /**
     * @param location classpath:some/dir (read-only, profiles shipped with the application),
     *                 file:some/dir or a plain directory path; unset means an in-memory store
     * @param required if true, creating a task without a matching profile fails, so a missing or
     *                 outdated profile stops the application at startup
     */
    public record Calibration(String location, @DefaultValue("false") boolean required) {}

    /**
     * @param enabled record dukehoff.decide.decisions, dukehoff.decide.decision.confidence and dukehoff.decide.gated when Micrometer
     *                is present
     */
    public record Metrics(@DefaultValue("true") boolean enabled) {}
}
