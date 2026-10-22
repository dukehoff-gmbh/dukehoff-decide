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

import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.State;
import com.dukehoff.decide.spring.DecideAutoConfigurationTest.Priority;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.ResolvableType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The starter against a real server, configured only through properties. Runs when
 * {@code DUKEHOFF_DECIDE_OPENAI_BASE_URL} and {@code DUKEHOFF_DECIDE_OPENAI_MODEL} are set.
 */
@EnabledIfEnvironmentVariable(named = "DUKEHOFF_DECIDE_OPENAI_BASE_URL", matches = ".+")
@EnabledIfEnvironmentVariable(named = "DUKEHOFF_DECIDE_OPENAI_MODEL", matches = ".+")
class DecideLiveTest {

    @Configuration(proxyBeanMethods = false)
    @EnableDecisionTasks(Priority.class)
    static class App {}

    @Test
    void decidesThroughAutoConfiguredBackend() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(DecideAutoConfiguration.class,
                        DecideHealthAutoConfiguration.class))
                .withUserConfiguration(App.class)
                .withPropertyValues("dukehoff.decide.openai.base-url=" + System.getenv("DUKEHOFF_DECIDE_OPENAI_BASE_URL"),
                        "dukehoff.decide.openai.model=" + System.getenv("DUKEHOFF_DECIDE_OPENAI_MODEL"))
                .run(ctx -> {
                    assertNull(ctx.getStartupFailure());
                    @SuppressWarnings("unchecked")
                    DecisionTask<Priority> task = (DecisionTask<Priority>) ctx.getBeanProvider(
                            ResolvableType.forClassWithGenerics(DecisionTask.class, Priority.class)).getObject();
                    var d = task.decide(State.of("Our production database is down and every customer gets errors!"));
                    System.out.println("distribution: " + d.distribution());
                    assertEquals(Priority.BLOCKER, d.value());

                    var health = ctx.getBean("dukehoffDecideHealthIndicator",
                            org.springframework.boot.health.contributor.HealthIndicator.class).health();
                    System.out.println("health: " + health.getStatus() + " " + health.getDetails());
                    assertEquals(org.springframework.boot.health.contributor.Status.UP, health.getStatus());
                });
    }
}
