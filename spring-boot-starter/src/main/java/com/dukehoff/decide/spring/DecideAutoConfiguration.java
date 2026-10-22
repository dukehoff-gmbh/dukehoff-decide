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

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.spi.ScoringBackend;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ResourceLoader;

import java.nio.file.Path;

/**
 * Creates, unless the application defines its own:
 * <ul>
 *   <li>an {@link OpenAiBackend} as {@link ScoringBackend}, if {@code dukehoff.decide.openai.model} is set</li>
 *   <li>a {@link CalibrationStore} from {@code dukehoff.decide.calibration.location}</li>
 *   <li>a {@link Decider}, if there is a backend</li>
 * </ul>
 * Declare {@link EnableDecisionTasks} to get injectable {@code DecisionTask<T>} beans.
 */
@AutoConfiguration
@EnableConfigurationProperties(DecideProperties.class)
public class DecideAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(ScoringBackend.class)
    @ConditionalOnProperty(prefix = "dukehoff.decide.openai", name = "model")
    public OpenAiBackend dukehoffDecideOpenAiBackend(DecideProperties properties) {
        DecideProperties.OpenAi p = properties.openai();
        OpenAiBackend.Builder b = OpenAiBackend.builder()
                .model(p.model())
                .topLogprobs(p.topLogprobs())
                .maxConcurrency(p.maxConcurrency())
                .maxRetries(p.maxRetries())
                .timeout(p.timeout());
        if (hasText(p.baseUrl())) b.baseUrl(p.baseUrl());
        if (hasText(p.apiKey())) b.apiKey(p.apiKey());
        if (hasText(p.modelId())) b.modelId(p.modelId());
        return b.build();
    }

    @Bean
    @ConditionalOnMissingBean
    public CalibrationStore dukehoffDecideCalibrationStore(DecideProperties properties, ResourceLoader resourceLoader) {
        String location = properties.calibration().location();
        if (!hasText(location)) return CalibrationStore.inMemory();
        if (location.startsWith("classpath:")) {
            ClassLoader cl = resourceLoader.getClassLoader();
            return CalibrationStore.classpath(location.substring("classpath:".length()),
                    cl != null ? cl : DecideAutoConfiguration.class.getClassLoader());
        }
        return CalibrationStore.directory(Path.of(location.startsWith("file:")
                ? location.substring("file:".length()) : location));
    }

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(ScoringBackend.class)
    public Decider dukehoffDecideDecider(ScoringBackend backend, CalibrationStore store, DecideProperties properties) {
        return Decider.builder()
                .backend(backend)
                .calibration(store)
                .requireCalibration(properties.calibration().required())
                .build();
    }

    private static boolean hasText(String s) {
        return s != null && !s.isBlank();
    }
}
