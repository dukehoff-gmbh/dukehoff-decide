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
package com.dukehoff.decide.embedding;

import java.util.List;

/**
 * Turns texts into fixed-length vectors with a frozen embedding model. {@link OpenAiEmbedder} calls an
 * OpenAI-compatible {@code /v1/embeddings} endpoint; implement this to use another runtime or to add a cache.
 */
public interface Embedder {

    /**
     * Identifies the embedding model. An {@link EmbeddingClassifier} only works with vectors from the model it was
     * trained on, so this is stored with it and checked on load.
     */
    String modelId();

    /** @return one vector per text, in the same order, all of the same length */
    List<float[]> embed(List<String> texts);

    default float[] embed(String text) {
        return embed(List.of(text)).getFirst();
    }
}
