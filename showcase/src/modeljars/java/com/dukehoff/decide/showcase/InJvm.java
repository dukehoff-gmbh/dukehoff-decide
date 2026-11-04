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
package com.dukehoff.decide.showcase;

import com.dukehoff.decide.embedding.Embedder;
import com.dukehoff.decide.modeljars.ModelsBackend;
import com.dukehoff.decide.modeljars.ModelsEmbedder;
import com.dukehoff.decide.spi.ScoringBackend;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;

import java.nio.file.Path;

/**
 * {@code -Dshowcase.engine=modeljars}: the model runs inside the JVM (dukehoff-decide-modeljars). Only compiled on
 * JDK 25+ (profile {@code modeljars}); {@link Showcase} loads it by name so the rest builds on Java 21. Run with
 * {@code MAVEN_OPTS="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED"}.
 */
public final class InJvm {

    private InJvm() {}

    /** {@code kernels}: {@code native} (Rust through FFM, default) or {@code java}. */
    public static ScoringBackend backend(String gguf, String modelId, String kernels) {
        Path path = Path.of(gguf);
        InferencePipeline pipeline = new InferencePipeline(
                kernels.equals("java") ? PureJavaBackend.load(path) : RustFfmBackend.load(path));
        return ModelsBackend.builder().pipeline(pipeline).modelId(modelId).build();
    }

    public static Embedder embedder(String gguf, String modelId) {
        return ModelsEmbedder.load(Path.of(gguf), modelId);
    }
}
