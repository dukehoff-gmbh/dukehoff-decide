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

import org.springframework.context.annotation.Import;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Registers one {@code DecisionTask<T>} bean per listed output type, created from the
 * {@link com.dukehoff.decide.Decider} bean. Tasks are created at startup, so an invalid output
 * type or (with {@code dukehoff.decide.calibration.required=true}) a missing calibration profile stops the
 * application right away instead of at the first request.
 *
 * <pre>
 * &#64;SpringBootApplication
 * &#64;EnableDecisionTasks({Triage.class, Sentiment.class})
 * class App {}
 *
 * &#64;Service
 * class TicketService {
 *     TicketService(DecisionTask&lt;Triage&gt; triage) { ... }
 * }
 * </pre>
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Import(DecisionTaskRegistrar.class)
public @interface EnableDecisionTasks {

    /** Output types: records, or enums annotated with {@code @Question}. */
    Class<?>[] value();
}
