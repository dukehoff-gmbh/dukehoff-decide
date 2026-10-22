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
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Outcome;
import com.dukehoff.decide.State;
import com.dukehoff.decide.spring.DecideAutoConfigurationTest.Priority;
import com.dukehoff.decide.spring.DecideAutoConfigurationTest.ScriptedBackendConfig;
import com.sun.net.httpserver.HttpServer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.boot.health.contributor.Status;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.ResolvableType;
import org.springframework.core.env.StandardEnvironment;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DecideObservabilityTest {

    final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withInitializer(ctx -> ctx.getEnvironment().getPropertySources()
                    .remove(StandardEnvironment.SYSTEM_ENVIRONMENT_PROPERTY_SOURCE_NAME))
            .withConfiguration(AutoConfigurations.of(DecideAutoConfiguration.class,
                    DecideHealthAutoConfiguration.class, DecideMetricsAutoConfiguration.class));

    @Configuration(proxyBeanMethods = false)
    @EnableDecisionTasks(Priority.class)
    static class PriorityTask {}

    @SuppressWarnings("unchecked")
    static DecisionTask<Priority> priorityTask(org.springframework.context.ApplicationContext ctx) {
        return (DecisionTask<Priority>) ctx.getBeanProvider(
                ResolvableType.forClassWithGenerics(DecisionTask.class, Priority.class)).getObject();
    }

    // ------------------------------------------------------------------------------------------ metrics

    @Test
    void recordsDecisionsConfidenceAndDeferrals() {
        runner.withUserConfiguration(ScriptedBackendConfig.class, PriorityTask.class)
                .withBean(MeterRegistry.class, SimpleMeterRegistry::new)
                .run(ctx -> {
                    assertNull(ctx.getStartupFailure());
                    DecisionTask<Priority> task = priorityTask(ctx);
                    task.decide(State.of("a"));
                    task.decideAll(List.of(State.of("b"), State.of("c")));
                    // the scripted backend gives 0.7 for BLOCKER: confident at 0.5, uncertain at 0.8
                    assertInstanceOf(Outcome.Confident.class, task.requireConfidence(0.5).decide(State.of("d")));
                    assertInstanceOf(Outcome.Uncertain.class, task.requireConfidence(0.8).decide(State.of("e")));
                    assertInstanceOf(Outcome.Uncertain.class,
                            task.requireConfidence(Map.of("value", 0.8)).decide(State.of("f")));

                    MeterRegistry r = ctx.getBean(MeterRegistry.class);
                    String taskId = Priority.class.getName();
                    assertEquals(4, r.get("dukehoff.decide.decisions").tag("task", taskId).tag("model", "scripted")
                            .tag("mode", "single").tag("outcome", "success").timer().count());
                    assertEquals(1, r.get("dukehoff.decide.decisions").tag("mode", "batch").timer().count());
                    var confidence = r.get("dukehoff.decide.decision.confidence").tag("task", taskId).summary();
                    assertEquals(6, confidence.count());
                    assertEquals(0.7, confidence.mean(), 1e-9);
                    assertEquals(1, r.get("dukehoff.decide.gated").tag("outcome", "confident").counter().count());
                    assertEquals(2, r.get("dukehoff.decide.gated").tag("outcome", "uncertain").counter().count());
                });
    }

    @Test
    void recordsErrors() {
        runner.withBean(com.dukehoff.decide.spi.ScoringBackend.class, () -> new com.dukehoff.decide.testing.ScriptedBackend(
                        "broken", (s, f) -> { throw new IllegalStateException("model down"); }))
                .withUserConfiguration(PriorityTask.class)
                .withBean(MeterRegistry.class, SimpleMeterRegistry::new)
                .run(ctx -> {
                    DecisionTask<Priority> task = priorityTask(ctx);
                    assertThrows(com.dukehoff.decide.DecideException.class, () -> task.decide(State.of("x")));
                    assertEquals(1, ctx.getBean(MeterRegistry.class).get("dukehoff.decide.decisions")
                            .tag("outcome", "error").timer().count());
                });
    }

    @Test
    void metricsCanBeSwitchedOffAndNeedARegistry() {
        runner.withUserConfiguration(ScriptedBackendConfig.class, PriorityTask.class)
                .withBean(MeterRegistry.class, SimpleMeterRegistry::new)
                .withPropertyValues("dukehoff.decide.metrics.enabled=false")
                .run(ctx -> {
                    priorityTask(ctx).decide(State.of("a"));
                    assertNull(ctx.getBean(MeterRegistry.class).find("dukehoff.decide.decisions").timer());
                    assertFalse(ctx.getBean(Decider.class) instanceof MeteredDecider);
                });

        runner.withUserConfiguration(ScriptedBackendConfig.class, PriorityTask.class).run(ctx -> {
            assertNull(ctx.getStartupFailure());
            assertFalse(ctx.getBean(Decider.class) instanceof MeteredDecider);
            assertEquals(Priority.BLOCKER, priorityTask(ctx).decide(State.of("a")).value());
        });
    }

    // ------------------------------------------------------------------------------------------- health

    @Test
    void healthIsUpWhenTheServerAnswers() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/models", exchange -> {
            byte[] body = "{\"data\":[{\"id\":\"qwen2.5-7b\"}]}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            runner.withPropertyValues("dukehoff.decide.openai.base-url=http://127.0.0.1:" + server.getAddress().getPort() + "/v1",
                            "dukehoff.decide.openai.model=qwen")
                    .run(ctx -> {
                        Health health = ctx.getBean("dukehoffDecideHealthIndicator", HealthIndicator.class).health();
                        assertEquals(Status.UP, health.getStatus());
                        assertEquals(List.of("qwen2.5-7b"), health.getDetails().get("serverModels"));
                        assertEquals("qwen", health.getDetails().get("model"));
                    });
        } finally {
            server.stop(0);
        }
    }

    @Test
    void healthIsDownWhenTheServerIsUnreachable() {
        runner.withPropertyValues("dukehoff.decide.openai.base-url=http://127.0.0.1:1/v1", "dukehoff.decide.openai.model=qwen")
                .run(ctx -> {
                    Health health = ctx.getBean("dukehoffDecideHealthIndicator", HealthIndicator.class).health();
                    assertEquals(Status.DOWN, health.getStatus());
                    assertEquals("http://127.0.0.1:1/v1", health.getDetails().get("baseUrl"));
                    assertNotNull(health.getDetails().get("error"), health.getDetails().toString());
                });
    }

    @Test
    void noHealthIndicatorForOtherBackendsOrWhenDisabled() {
        runner.withUserConfiguration(ScriptedBackendConfig.class)
                .run(ctx -> assertFalse(ctx.containsBean("dukehoffDecideHealthIndicator")));
        runner.withPropertyValues("dukehoff.decide.openai.model=qwen", "management.health.dukehoff-decide.enabled=false")
                .run(ctx -> assertFalse(ctx.containsBean("dukehoffDecideHealthIndicator")));
    }
}
