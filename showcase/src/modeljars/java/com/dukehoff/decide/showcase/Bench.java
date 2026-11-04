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

import com.dukehoff.decide.modeljars.ModelsBackend;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.showcase.ShowcaseDataset.Example;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.spi.TaskSchema;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Latency with one request in flight, uncached: {@code -Dbench.engine=openai|native|java}, {@code -Dbench.n} examples
 * after {@code -Dbench.warmup}, {@code -Dbench.reuse} (in-JVM prefix reuse), dataset options as for {@link Showcase}.
 * Prints one Markdown table row. Times are per field: the wall time of an example divided by its fields.
 */
public final class Bench {

    public static void main(String[] args) throws Exception {
        Showcase.Config cfg = Showcase.Config.fromSystemProperties();
        run(ShowcaseDataset.byId(cfg.dataset()), cfg);
    }

    private static <T> void run(ShowcaseDataset<T> dataset, Showcase.Config cfg) throws Exception {
        String engine = System.getProperty("bench.engine", "native");
        int n = Integer.getInteger("bench.n", 30), warmup = Integer.getInteger("bench.warmup", 3);
        boolean reuse = Boolean.parseBoolean(System.getProperty("bench.reuse", "true"));
        List<Example<T>> examples = ShowcaseDataset.sample(dataset.load(cfg.cache().resolve("datasets")), warmup + n,
                cfg.seed(), cfg.language());
        TaskSchema schema = TaskSchema.of(dataset.type());

        InferencePipeline pipeline = null;
        ScoringBackend backend;
        if (engine.equals("openai")) {
            backend = OpenAiBackend.builder().baseUrl(cfg.baseUrl()).model(cfg.model()).maxConcurrency(1)
                    .timeout(Duration.ofMinutes(5)).build();
        } else {
            Path gguf = Path.of(System.getProperty("showcase.gguf"));
            pipeline = new InferencePipeline(engine.equals("java") ? PureJavaBackend.load(gguf) : RustFfmBackend.load(gguf));
            backend = ModelsBackend.builder().pipeline(pipeline).modelId(cfg.model() + "@" + engine).reusePrefix(reuse).build();
        }

        int fields = schema.fields().size();
        double[] ms = new double[n];
        long tokens = 0;
        for (int i = 0; i < warmup + n; i++) {
            ScoringRequest request = new ScoringRequest(examples.get(i).state(), schema);
            long start = System.nanoTime();
            backend.score(request);
            double each = (System.nanoTime() - start) / 1e6 / fields;
            if (i < warmup) continue;
            ms[i - warmup] = each;
            if (pipeline != null) {
                for (FieldSpec f : schema.fields()) {
                    LabelPrompt p = LabelPrompt.of(schema, f, request.state());
                    tokens += pipeline.tokenize(ChatTemplate.CHATML.render(List.of(ChatMessage.system(p.system()),
                            ChatMessage.user(p.user())))).length;
                }
            }
        }
        if (backend instanceof AutoCloseable c) c.close();
        Arrays.sort(ms);
        System.out.printf(Locale.ROOT, "| %s %s | %s | %d | %s | %.0f | %.0f | %s |%n", cfg.dataset(), cfg.language(),
                engine.equals("openai") ? cfg.baseUrl() : engine + (reuse ? ", prefix reuse" : ", full prefill"),
                fields, System.getProperty("bench.threads", "default"), ms[n / 2], ms[(int) (0.95 * (n - 1))],
                pipeline == null ? "" : String.format(Locale.ROOT, "%.0f", (double) tokens / n / fields));
    }
}
