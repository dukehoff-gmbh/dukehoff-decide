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
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;

/**
 * Micrometer metrics for every {@link Decider} bean (the auto-configured one or your own), and so
 * for every task it creates. See {@link MeteredDecisionTask} for the meters. Active with Micrometer
 * on the classpath and a {@link MeterRegistry} bean; switch off with {@code dukehoff.decide.metrics.enabled=false}.
 */
@AutoConfiguration(after = DecideAutoConfiguration.class)
@ConditionalOnClass(MeterRegistry.class)
@ConditionalOnProperty(prefix = "dukehoff.decide.metrics", name = "enabled", matchIfMissing = true)
public class DecideMetricsAutoConfiguration {

    /** Static: bean post-processors are created before regular beans. The registry is looked up lazily. */
    @Bean
    static BeanPostProcessor dukehoffDecideMeteredDeciderPostProcessor(ObjectProvider<MeterRegistry> registry) {
        return new BeanPostProcessor() {
            @Override
            public Object postProcessAfterInitialization(Object bean, String beanName) {
                if (bean instanceof Decider decider && !(bean instanceof MeteredDecider)) {
                    MeterRegistry r = registry.getIfAvailable();
                    if (r != null) return new MeteredDecider(decider, r);
                }
                return bean;
            }
        };
    }
}
