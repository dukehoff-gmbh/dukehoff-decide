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
package com.dukehoff.decide.openai;

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.State;
import com.dukehoff.decide.openai.OpenAiBackendTest.Priority;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Against a real server. Runs only when {@code DUKEHOFF_DECIDE_OPENAI_BASE_URL} and {@code DUKEHOFF_DECIDE_OPENAI_MODEL}
 * are set ({@code DUKEHOFF_DECIDE_OPENAI_API_KEY} is optional), e.g. for llama.cpp:
 * <pre>
 * DUKEHOFF_DECIDE_OPENAI_BASE_URL=http://localhost:8080/v1 DUKEHOFF_DECIDE_OPENAI_MODEL=qwen2.5-3b-instruct mvn verify
 * </pre>
 */
@EnabledIfEnvironmentVariable(named = "DUKEHOFF_DECIDE_OPENAI_BASE_URL", matches = ".+")
@EnabledIfEnvironmentVariable(named = "DUKEHOFF_DECIDE_OPENAI_MODEL", matches = ".+")
class OpenAiBackendLiveTest {

    @Test
    void obviousTicketIsABlocker() {
        OpenAiBackend backend = OpenAiBackend.builder()
                .baseUrl(System.getenv("DUKEHOFF_DECIDE_OPENAI_BASE_URL"))
                .model(System.getenv("DUKEHOFF_DECIDE_OPENAI_MODEL"))
                .apiKey(System.getenv("DUKEHOFF_DECIDE_OPENAI_API_KEY"))
                .build();
        Decision<Priority> d = Decider.builder().backend(backend).build().task(Priority.class)
                .decide(State.of("Our production database is down and every customer gets a 500 error. Fix it now!"));
        System.out.println("distribution: " + d.distribution(Decision.VALUE));
        assertEquals(Priority.BLOCKER, d.value());
    }
}
