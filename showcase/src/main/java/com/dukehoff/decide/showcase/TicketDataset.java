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

import com.dukehoff.decide.State;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * {@code Tobi-Bueck/customer-support-tickets} on Hugging Face (CC BY-NC 4.0), 20k synthetic tickets
 * in English and German, mapped to {@link Triage}. Many labels cannot be inferred from the ticket
 * text, which makes it a good test of whether calibration catches a confidently wrong model.
 */
final class TicketDataset implements ShowcaseDataset<Triage> {

    static final String SOURCE = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets";
    static final String CSV_URL = SOURCE + "/resolve/main/dataset-tickets-multi-lang-4-20k.csv";

    @Override
    public String id() {
        return "tickets";
    }

    @Override
    public String title() {
        return "support ticket triage";
    }

    @Override
    public String citation() {
        return "[Tobi-Bueck/customer-support-tickets](" + SOURCE + ") (CC BY-NC 4.0, synthetic, EN/DE)";
    }

    @Override
    public Class<Triage> type() {
        return Triage.class;
    }

    @Override
    public Map<String, Function<Triage, Object>> fields() {
        Map<String, Function<Triage, Object>> f = new LinkedHashMap<>();
        f.put("type", Triage::type);
        f.put("queue", Triage::queue);
        f.put("priority", Triage::priority);
        return f;
    }

    @Override
    public List<Example<Triage>> load(Path cacheDir) {
        Path csv = ShowcaseDataset.download(CSV_URL, cacheDir.resolve("customer-support-tickets-4-20k.csv"));
        try {
            return parse(Csv.read(Files.readString(csv, StandardCharsets.UTF_8)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static List<Example<Triage>> parse(List<Map<String, String>> rows) {
        List<Example<Triage>> out = new ArrayList<>(rows.size());
        for (Map<String, String> r : rows) {
            String body = r.getOrDefault("body", "");
            if (body.isBlank()) continue;
            State.Builder state = State.builder();
            String subject = r.getOrDefault("subject", "");
            if (!subject.isBlank()) state.put("Subject", subject);
            out.add(new Example<>(state.text(body).build(), new Triage(
                    ShowcaseDataset.constant(Triage.Type.class, r.get("type")),
                    ShowcaseDataset.constant(Triage.Queue.class, r.get("queue")),
                    ShowcaseDataset.constant(Triage.Priority.class, r.get("priority"))), r.get("language")));
        }
        return out;
    }
}
