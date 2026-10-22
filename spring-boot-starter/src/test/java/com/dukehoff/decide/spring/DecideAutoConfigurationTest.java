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
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.calibration.Calibration;
import com.dukehoff.decide.calibration.CalibrationStore;
import com.dukehoff.decide.calibration.Calibrator;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.ResolvableType;
import org.springframework.core.env.StandardEnvironment;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static com.dukehoff.decide.testing.ScriptedBackend.probabilities;
import static org.junit.jupiter.api.Assertions.*;

class DecideAutoConfigurationTest {

    @Question("How urgent is this ticket?")
    enum Priority {
        @Option("Production is down") BLOCKER,
        @Option("Something is broken") NORMAL,
        @Option("A question or wish") LOW
    }

    record Ticket(@Question("How urgent is it?") Priority priority,
                  @Question("Does the customer threaten to leave?") boolean churnRisk) {}

    @Configuration(proxyBeanMethods = false)
    static class ScriptedBackendConfig {
        @Bean
        ScoringBackend backend() {
            return new ScriptedBackend("scripted", (state, field) -> field.options().size() == 3
                    ? probabilities(0.7, 0.2, 0.1)
                    : probabilities(0.1, 0.9));
        }
    }

    @Configuration(proxyBeanMethods = false)
    @EnableDecisionTasks({Priority.class, Ticket.class})
    static class Tasks {}

    record TicketService(DecisionTask<Priority> priority, DecisionTask<Ticket> ticket) {}

    @Configuration(proxyBeanMethods = false)
    static class ServiceConfig {
        @Bean
        TicketService ticketService(DecisionTask<Priority> priority, DecisionTask<Ticket> ticket) {
            return new TicketService(priority, ticket);
        }
    }

    /**
     * Without the process environment: DUKEHOFF_DECIDE_OPENAI_MODEL (set for the live tests) would otherwise bind
     * to dukehoff.decide.openai.model and create a backend here.
     */
    final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withInitializer(ctx -> ctx.getEnvironment().getPropertySources()
                    .remove(StandardEnvironment.SYSTEM_ENVIRONMENT_PROPERTY_SOURCE_NAME))
            .withConfiguration(AutoConfigurations.of(DecideAutoConfiguration.class));

    @Test
    void createsNoBackendWithoutConfiguration() {
        runner.run(ctx -> {
            assertNull(ctx.getStartupFailure());
            assertTrue(ctx.getBeansOfType(ScoringBackend.class).isEmpty());
            assertTrue(ctx.getBeansOfType(Decider.class).isEmpty());
            assertNotNull(ctx.getBean(CalibrationStore.class));
        });
    }

    @Test
    void createsOpenAiBackendFromProperties() {
        runner.withPropertyValues(
                        "dukehoff.decide.openai.base-url=http://localhost:1/v1",
                        "dukehoff.decide.openai.model=qwen2.5-7b-instruct",
                        "dukehoff.decide.openai.model-id=qwen2.5-7b-instruct-q4_k_m",
                        "dukehoff.decide.openai.timeout=5s")
                .run(ctx -> {
                    assertNull(ctx.getStartupFailure());
                    OpenAiBackend backend = ctx.getBean(OpenAiBackend.class);
                    assertEquals("qwen2.5-7b-instruct-q4_k_m", backend.modelId());
                    assertNotNull(ctx.getBean(Decider.class));
                });
    }

    @Test
    void ownBackendWins() {
        runner.withUserConfiguration(ScriptedBackendConfig.class)
                .withPropertyValues("dukehoff.decide.openai.model=ignored")
                .run(ctx -> {
                    assertEquals("scripted", ctx.getBean(ScoringBackend.class).modelId());
                    assertTrue(ctx.getBeansOfType(OpenAiBackend.class).isEmpty());
                });
    }

    @Test
    void injectsTypedDecisionTasks() {
        runner.withUserConfiguration(ScriptedBackendConfig.class, Tasks.class, ServiceConfig.class).run(ctx -> {
            assertNull(ctx.getStartupFailure());
            TicketService service = ctx.getBean(TicketService.class);
            assertEquals(Priority.BLOCKER, service.priority().decide(State.of("Server down")).value());
            assertEquals(new Ticket(Priority.BLOCKER, false), service.ticket().decide(State.of("Server down")).value());

            var type = ResolvableType.forClassWithGenerics(DecisionTask.class, Priority.class);
            assertEquals(1, ctx.getBeanNamesForType(type).length);
        });
    }

    @Test
    void loadsRequiredCalibrationFromClasspath() {
        // src/test/resources/decide/calibration holds a profile for Priority on "scripted" with T = 2
        runner.withUserConfiguration(ScriptedBackendConfig.class, Tasks.class)
                .withPropertyValues("dukehoff.decide.calibration.location=classpath:decide/calibration",
                        "dukehoff.decide.calibration.required=true")
                .run(ctx -> {
                    // Ticket has no profile, so with required=true the context must not start
                    assertNotNull(ctx.getStartupFailure());
                    assertTrue(rootMessage(ctx.getStartupFailure()).contains(Ticket.class.getName()),
                            rootMessage(ctx.getStartupFailure()));
                });

        runner.withUserConfiguration(ScriptedBackendConfig.class, PriorityOnly.class)
                .withPropertyValues("dukehoff.decide.calibration.location=classpath:decide/calibration",
                        "dukehoff.decide.calibration.required=true")
                .run(ctx -> {
                    assertNull(ctx.getStartupFailure());
                    @SuppressWarnings("unchecked")
                    DecisionTask<Priority> task = (DecisionTask<Priority>) ctx.getBeanProvider(
                            ResolvableType.forClassWithGenerics(DecisionTask.class, Priority.class)).getObject();
                    Decision<Priority> d = task.decide(State.of("Server down"));
                    assertTrue(d.calibrated());
                    double[] softened = Calibrator.temperature(2).apply(probabilities(0.7, 0.2, 0.1));
                    assertEquals(softened[0], d.confidence(), 1e-9);
                });
    }

    @Configuration(proxyBeanMethods = false)
    @EnableDecisionTasks(Priority.class)
    static class PriorityOnly {}

    @Test
    void writesProfilesToADirectory(@TempDir Path dir) {
        runner.withPropertyValues("dukehoff.decide.calibration.location=file:" + dir).run(ctx -> {
            ctx.getBean(CalibrationStore.class).save(
                    new Calibration.Profile("t", "m", Map.of("f", Calibrator.temperature(1.5))));
            try (var files = Files.list(dir)) {
                assertEquals(1, files.count());
            }
        });
    }

    @Test
    void tasksWithoutBackendExplainWhatIsMissing() {
        runner.withUserConfiguration(Tasks.class).run(ctx -> {
            assertNotNull(ctx.getStartupFailure());
            assertTrue(rootMessage(ctx.getStartupFailure()).contains("dukehoff.decide.openai.model"),
                    rootMessage(ctx.getStartupFailure()));
        });
    }

    private static String rootMessage(Throwable t) {
        while (t.getCause() != null) t = t.getCause();
        return t.getMessage();
    }
}
