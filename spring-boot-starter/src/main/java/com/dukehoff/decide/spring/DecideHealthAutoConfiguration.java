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

import com.dukehoff.decide.openai.OpenAiBackend;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.health.autoconfigure.contributor.ConditionalOnEnabledHealthIndicator;
import org.springframework.boot.health.contributor.AbstractHealthIndicator;
import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.context.annotation.Bean;

import java.time.Duration;
import java.util.List;

/**
 * Health component {@code dukehoffDecide}: whether the OpenAI-compatible server answers {@code GET /models}.
 * Active with Actuator on the classpath and an {@link OpenAiBackend} bean; switch off with
 * {@code management.health.dukehoff-decide.enabled=false}.
 */
@AutoConfiguration(after = DecideAutoConfiguration.class)
@ConditionalOnClass(HealthIndicator.class)
@ConditionalOnBean(OpenAiBackend.class)
@ConditionalOnEnabledHealthIndicator("dukehoff-decide")
public class DecideHealthAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(name = "dukehoffDecideHealthIndicator")
    public HealthIndicator dukehoffDecideHealthIndicator(OpenAiBackend backend) {
        return new OpenAiHealthIndicator(backend, Duration.ofSeconds(5));
    }

    static final class OpenAiHealthIndicator extends AbstractHealthIndicator {

        private final OpenAiBackend backend;
        private final Duration timeout;

        OpenAiHealthIndicator(OpenAiBackend backend, Duration timeout) {
            super("dukehoff-decide backend check failed");
            this.backend = backend;
            this.timeout = timeout;
        }

        @Override
        protected void doHealthCheck(Health.Builder builder) {
            builder.withDetail("baseUrl", backend.baseUrl())
                    .withDetail("model", backend.model())
                    .withDetail("modelId", backend.modelId());
            List<String> models = backend.models(timeout); // throws -> DOWN with the exception as detail
            builder.up().withDetail("serverModels", models);
        }
    }
}
