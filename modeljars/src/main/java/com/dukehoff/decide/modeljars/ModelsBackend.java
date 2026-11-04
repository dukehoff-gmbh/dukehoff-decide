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

import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Scores {@link LabelPrompt}s with a GGUF model running inside the JVM
 * (<a href="https://github.com/integrallis/models">Integrallis Models</a>), no model server needed.
 *
 * <pre>{@code
 * try (ModelsBackend backend = ModelsBackend.builder()
 *         .model(Path.of("qwen2.5-1.5b-instruct-q4_k_m.gguf"))
 *         .modelId("qwen2.5-1.5b-instruct-q4_k_m")
 *         .build()) {
 *     Decider decider = Decider.builder().backend(backend).calibration(store).build();
 * }
 * }</pre>
 *
 * Each field is one prefill of the rendered chat prompt; the scores are the log-softmax over the whole
 * vocabulary at the answer position, summed (log-sum-exp) over every token that reads as the label, such as
 * {@code "A"} and {@code " A"}. Unlike an OpenAI-style top-k list, no label is ever missing. The model runs
 * one prompt at a time; requests are scored in order. The tokens a prompt shares with the previous one (the
 * system prompt, and the input when a task has several fields) are kept in the context and not prefilled again,
 * if the backend can rewind.
 */
public final class ModelsBackend implements ScoringBackend, AutoCloseable {

    private final InferencePipeline pipeline;
    private final ChatTemplate template;
    private final String modelId;
    /** Token ids per label A..Z: every vocabulary entry that normalises to that letter. */
    private final int[][] labelTokens;
    private final boolean reusePrefix;
    /** The tokens in the context, or empty after a failure. */
    private int[] context = new int[0];

    private ModelsBackend(InferencePipeline pipeline, ChatTemplate template, String modelId, boolean reusePrefix) {
        this.pipeline = pipeline;
        this.template = template;
        this.modelId = modelId;
        this.labelTokens = labelTokens(pipeline.tokenizer());
        this.reusePrefix = reusePrefix && pipeline.rewindable();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        Map<String, RawScores> fields = new LinkedHashMap<>();
        for (FieldSpec f : request.schema().fields()) {
            fields.put(f.name(), score(LabelPrompt.of(request.schema(), f, request.state())));
        }
        return fields;
    }

    private synchronized RawScores score(LabelPrompt prompt) {
        ModelPrompt rendered = template.render(List.of(ChatMessage.system(prompt.system()), ChatMessage.user(prompt.user())));
        float[] logits = prefill(pipeline.tokenize(rendered));
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        double sum = 0;
        for (float v : logits) sum += Math.exp(v - max);
        double logZ = max + Math.log(sum);

        double[] out = new double[prompt.labels().size()];
        for (int i = 0; i < out.length; i++) {
            int[] ids = labelTokens[prompt.labels().get(i).charAt(0) - 'A'];
            double m = Double.NEGATIVE_INFINITY;
            for (int id : ids) m = Math.max(m, logits[id]);
            double s = 0;
            for (int id : ids) s += Math.exp(logits[id] - m);
            out[i] = ids.length == 0 ? Double.NEGATIVE_INFINITY : m + Math.log(s) - logZ;
        }
        return new RawScores(out);
    }

    /** Prefills {@code tokens}, keeping the prefix shared with the context; at least one token is run for the logits. */
    private float[] prefill(int[] tokens) {
        int keep = reusePrefix ? Math.min(sharedPrefix(context, tokens), tokens.length - 1) : 0;
        int[] previous = context;
        context = new int[0];
        if (keep > 0) pipeline.rewind(keep);
        else pipeline.resetContext();
        float[] logits = pipeline.prefill(keep == 0 ? tokens : Arrays.copyOfRange(tokens, keep, tokens.length), keep);
        context = tokens;
        return logits;
    }

    static int sharedPrefix(int[] a, int[] b) {
        int n = Math.min(a.length, b.length), i = 0;
        while (i < n && a[i] == b[i]) i++;
        return i;
    }

    static int[][] labelTokens(Tokenizer tokenizer) {
        List<List<Integer>> byLabel = new ArrayList<>();
        for (int i = 0; i < LabelPrompt.MAX_OPTIONS; i++) byLabel.add(new ArrayList<>());
        for (int id = 0; id < tokenizer.vocabSize(); id++) {
            String label = normalise(tokenizer.decode(id));
            if (label.length() == 1 && label.charAt(0) >= 'A' && label.charAt(0) <= 'Z') byLabel.get(label.charAt(0) - 'A').add(id);
        }
        return byLabel.stream().map(ids -> ids.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
    }

    /** Same rule as {@link LabelPrompt#scores}: strip leading and trailing characters that are not letters or digits. */
    private static String normalise(String token) {
        int start = 0, end = token.length();
        while (start < end && !Character.isLetterOrDigit(token.charAt(start))) start++;
        while (end > start && !Character.isLetterOrDigit(token.charAt(end - 1))) end--;
        return token.substring(start, end);
    }

    /** Closes the model (the mapped GGUF file and its native resources). */
    @Override
    public synchronized void close() {
        pipeline.close();
    }

    @Override
    public String toString() {
        return modelId;
    }

    public static final class Builder {
        private InferencePipeline pipeline;
        private ChatTemplate template = ChatTemplate.CHATML;
        private String modelId;
        private String fileModelId;
        private boolean reusePrefix = true;

        private Builder() {}

        /**
         * Loads a GGUF file with the pure Java backend. The model id defaults to the file name without
         * {@code .gguf}, which usually names the quantisation ({@code qwen2.5-1.5b-instruct-q4_k_m}).
         */
        public Builder model(Path gguf) {
            pipeline(new InferencePipeline(PureJavaBackend.load(gguf)));
            fileModelId = gguf.getFileName().toString().replaceFirst("(?i)\\.gguf$", "");
            return this;
        }

        /** A pipeline loaded elsewhere; the backend owns it from now on and closes it. */
        public Builder pipeline(InferencePipeline pipeline) {
            this.pipeline = Objects.requireNonNull(pipeline);
            fileModelId = null;
            return this;
        }

        /** Chat template of the model; default {@link ChatTemplate#CHATML} (Qwen2.5). */
        public Builder template(ChatTemplate template) { this.template = Objects.requireNonNull(template); return this; }

        /**
         * Calibration profiles are keyed by it, so it must change with the weights and their quantisation: a
         * profile fitted on Q8_0 does not fit Q4_K_M. Required with {@link #pipeline}, because the GGUF metadata
         * name does not say which quantisation was loaded.
         */
        public Builder modelId(String modelId) { this.modelId = Objects.requireNonNull(modelId); return this; }

        /** Keep the prompt prefix shared with the previous prompt in the context; default true. */
        public Builder reusePrefix(boolean reusePrefix) { this.reusePrefix = reusePrefix; return this; }

        public ModelsBackend build() {
            if (pipeline == null) throw new IllegalStateException("model(Path) or pipeline(...) is required");
            String id = modelId != null ? modelId : fileModelId;
            if (id == null) {
                throw new IllegalStateException("modelId(...) is required with pipeline(...): name the model and its "
                        + "quantisation, e.g. qwen2.5-1.5b-instruct-q4_k_m");
            }
            return new ModelsBackend(pipeline, template, id, reusePrefix);
        }
    }
}
