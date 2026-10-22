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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The unstructured input: free text plus optional named attributes.
 * Deliberately not a chat message list - it describes a thing, not a conversation.
 * Attribute order is preserved, because it is also the order in which the model sees them.
 */
public record State(String text, Map<String, String> attributes) {

    public State {
        Objects.requireNonNull(text, "text");
        attributes = Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
    }

    public static State of(String text) {
        return new State(text, Map.of());
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Renders the state as plain text, attributes first. Backends may use their own format. */
    public String render() {
        StringBuilder sb = new StringBuilder();
        attributes.forEach((k, v) -> sb.append(k).append(": ").append(v).append('\n'));
        if (!text.isBlank()) {
            if (!attributes.isEmpty()) sb.append('\n');
            sb.append(text);
        }
        return sb.toString().strip();
    }

    public static final class Builder {
        private String text = "";
        private final Map<String, String> attributes = new LinkedHashMap<>();

        public Builder text(String text) { this.text = text; return this; }
        public Builder put(String name, Object value) { attributes.put(name, String.valueOf(value)); return this; }
        public State build() { return new State(text, attributes); }
    }
}
