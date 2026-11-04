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
package com.dukehoff.decide.modeljars;

import com.dukehoff.decide.embedding.Embedder;
import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * An {@link Embedder} that runs a GGUF embedding model inside the JVM
 * (<a href="https://github.com/integrallis/models">Integrallis Models</a>), for
 * {@link com.dukehoff.decide.embedding.EmbeddingClassifier}. Needs {@code dukehoff-decide-embedding} on the
 * class path.
 *
 * <pre>{@code
 * try (ModelsEmbedder embedder = ModelsEmbedder.load(Path.of("Qwen3-Embedding-0.6B-Q8_0.gguf"), "qwen3-embedding-0.6b-q8_0")) {
 *     EmbeddingClassifier classifier = EmbeddingClassifier.trainer(embedder).train(Triage.class, labelled);
 * }
 * }</pre>
 */
public final class ModelsEmbedder implements Embedder, AutoCloseable {

    private final EmbeddingBackend backend;
    private final String modelId;

    public ModelsEmbedder(EmbeddingBackend backend, String modelId) {
        this.backend = Objects.requireNonNull(backend);
        this.modelId = Objects.requireNonNull(modelId);
    }

    /** Loads a GGUF file with the pure Java backend and the pooling the model declares (or last token). */
    public static ModelsEmbedder load(Path gguf, String modelId) {
        return new ModelsEmbedder(GgufEmbeddingBackend.builder(PureJavaBackend.load(gguf)).build(), modelId);
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public synchronized List<float[]> embed(List<String> texts) {
        return Arrays.asList(backend.embedAll(texts));
    }

    @Override
    public synchronized void close() {
        backend.close();
    }
}
