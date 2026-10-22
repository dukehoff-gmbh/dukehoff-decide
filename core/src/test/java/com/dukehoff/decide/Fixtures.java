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

/** Output types shared by the tests. */
final class Fixtures {

    private Fixtures() {}

    @Question("How urgent is this ticket?")
    enum Priority {
        @Option("Production is down or data is being lost") BLOCKER,
        @Option("A feature is broken for many users, no workaround") HIGH,
        @Option("Something is broken but there is a workaround") NORMAL,
        @Option("Question, cosmetic issue or feature wish") LOW
    }

    enum Team { BILLING, PLATFORM, MOBILE, OTHER }

    @Question("Triage an incoming customer support ticket")
    record Triage(
            @Question("How urgent is this ticket?") Priority priority,
            @Question("Which team should handle it?") Team team,
            @Question("Does the customer threaten to cancel or leave?") boolean churnRisk,
            @Question("How angry is the customer?")
            @Scale(min = 0, max = 1, steps = 5, anchors = {"calm", "furious"}) double anger) {}

    record Stars(@Question("How many stars would the reviewer give?") @Scale(min = 1, max = 5) int stars) {}

    record Validated(@Question("Is it spam?") boolean spam) {
        Validated {
            if (spam) throw new IllegalArgumentException("spam not allowed here");
        }
    }

    record MissingQuestion(Priority priority) {}

    record UnscaledNumber(@Question("How many?") int count) {}

    record UnsupportedType(@Question("Which name?") String name) {}

    record BadAnchors(@Question("How much?") @Scale(steps = 5, anchors = {"a", "b", "c"}) double x) {}

    enum NoQuestion { A, B }
}
