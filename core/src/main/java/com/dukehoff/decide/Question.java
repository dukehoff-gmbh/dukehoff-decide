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
package com.dukehoff.decide;

import java.lang.annotation.*;

/**
 * The natural-language question the model answers for a record component.
 * On the record type itself it acts as shared context/instructions for all fields.
 *
 * <pre>
 * &#64;Question("Triage an incoming support ticket")
 * record Triage(
 *     &#64;Question("How urgent is this ticket?") Priority priority,
 *     &#64;Question("Does the customer threaten to cancel?") boolean churnRisk) {}
 * </pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.RECORD_COMPONENT, ElementType.TYPE})
public @interface Question {
    String value();
}
