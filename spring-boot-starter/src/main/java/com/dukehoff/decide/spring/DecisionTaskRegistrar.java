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

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.Decider;
import com.dukehoff.decide.DecisionTask;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.BeanFactoryAware;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.context.annotation.ImportBeanDefinitionRegistrar;
import org.springframework.core.ResolvableType;
import org.springframework.core.annotation.AnnotationAttributes;
import org.springframework.core.type.AnnotationMetadata;

/**
 * Registers a bean of type {@code DecisionTask<T>} (with the generic type set, so it can be
 * injected by type) for every class listed in {@link EnableDecisionTasks}.
 */
class DecisionTaskRegistrar implements ImportBeanDefinitionRegistrar, BeanFactoryAware {

    private BeanFactory beanFactory;

    @Override
    public void setBeanFactory(BeanFactory beanFactory) {
        this.beanFactory = beanFactory;
    }

    @Override
    public void registerBeanDefinitions(AnnotationMetadata metadata, BeanDefinitionRegistry registry) {
        AnnotationAttributes attrs = AnnotationAttributes.fromMap(
                metadata.getAnnotationAttributes(EnableDecisionTasks.class.getName()));
        if (attrs == null) return;
        for (Class<?> type : attrs.getClassArray("value")) {
            String name = "dukehoffDecideDecisionTask." + type.getName();
            if (registry.containsBeanDefinition(name)) continue;
            RootBeanDefinition bd = new RootBeanDefinition(DecisionTask.class);
            bd.setTargetType(ResolvableType.forClassWithGenerics(DecisionTask.class, type));
            bd.setInstanceSupplier(() -> decider().task(type));
            registry.registerBeanDefinition(name, bd);
        }
    }

    private Decider decider() {
        Decider decider = beanFactory.getBeanProvider(Decider.class).getIfAvailable();
        if (decider == null) {
            throw new DecideException("@EnableDecisionTasks needs a Decider: set dukehoff.decide.openai.model, "
                    + "or define a ScoringBackend or Decider bean");
        }
        return decider;
    }
}
