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
package com.dukehoff.decide.spi;

import com.dukehoff.decide.State;

import java.util.List;
import java.util.Map;

/**
 * The only thing a model integration has to implement.
 *
 * <p>Typical implementations:
 * <ul>
 *   <li>Logprob backends (llama.cpp, Ollama, vLLM, OpenAI): prompt per field (or one prompt for all
 *       fields), read the log-probabilities of the option labels as the next token.</li>
 *   <li>Trained classifiers: {@code LinearClassifier} in {@code dukehoff-decide-classifier} (TF-IDF + logistic
 *       regression per field), or a fine-tuned model with one classification head per field.</li>
 *   <li>{@link com.dukehoff.decide.testing.ScriptedBackend} for unit tests.</li>
 * </ul>
 * Backends return raw scores; calibration happens above this layer so that backends stay dumb.
 */
public interface ScoringBackend {

    /** Identifies model + version; calibration profiles are only valid for the same id. */
    String modelId();

    /** @return raw scores for every field in {@code request.schema().fields()}, keyed by field name */
    Map<String, RawScores> score(ScoringRequest request);

    /** Override to batch. Order of results must match order of requests. */
    default List<Map<String, RawScores>> scoreAll(List<ScoringRequest> requests) {
        return requests.stream().map(this::score).toList();
    }

    record ScoringRequest(State state, TaskSchema schema) {}

    /**
     * Log-probabilities (or any logits) over {@link FieldSpec#options()}, same order.
     * Need not be normalised; calibration applies a softmax.
     */
    record RawScores(double[] logProbs) {
        public RawScores {
            logProbs = logProbs.clone();
        }

        public static RawScores of(double... logProbs) {
            return new RawScores(logProbs);
        }
    }
}
