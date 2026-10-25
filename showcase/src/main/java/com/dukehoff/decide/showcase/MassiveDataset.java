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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.zip.GZIPInputStream;

/**
 * Amazon MASSIVE 1.1 (CC BY 4.0): about 16,500 crowd-written voice assistant requests per language, mapped to
 * {@link AssistantRequest}. Only en-US and de-DE are loaded. The German rows are translations of the English
 * ones, so run one language at a time ({@code -Dshowcase.language=en} or {@code de}).
 */
final class MassiveDataset implements ShowcaseDataset<AssistantRequest> {

    static final String SOURCE = "https://github.com/alexa/massive";
    static final String TAR_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz";
    static final Map<String, String> LOCALES = Map.of("1.1/data/en-US.jsonl", "en", "1.1/data/de-DE.jsonl", "de");

    private static final ObjectMapper JSON = new ObjectMapper();

    @Override
    public String id() {
        return "massive";
    }

    @Override
    public String title() {
        return "voice assistant routing";
    }

    @Override
    public String citation() {
        return "[Amazon MASSIVE 1.1](" + SOURCE + ") (CC BY 4.0, EN and DE; crowd-written, German translated from English)";
    }

    @Override
    public Class<AssistantRequest> type() {
        return AssistantRequest.class;
    }

    @Override
    public Map<String, Function<AssistantRequest, Object>> fields() {
        Map<String, Function<AssistantRequest, Object>> f = new LinkedHashMap<>();
        f.put("scenario", AssistantRequest::scenario);
        return f;
    }

    @Override
    public List<Example<AssistantRequest>> load(Path cacheDir) {
        Path tar = ShowcaseDataset.download(TAR_URL, cacheDir.resolve("amazon-massive-dataset-1.1.tar.gz"));
        List<Example<AssistantRequest>> out = new ArrayList<>();
        try (InputStream in = new GZIPInputStream(Files.newInputStream(tar))) {
            byte[] header = new byte[512];
            while (in.readNBytes(header, 0, 512) == 512 && header[0] != 0) {
                String name = field(header, 345, 155).isEmpty() ? field(header, 0, 100)
                        : field(header, 345, 155) + '/' + field(header, 0, 100);
                long size = Long.parseLong(field(header, 124, 12).trim(), 8);
                byte[] body = in.readNBytes((int) size);
                in.skipNBytes((512 - size % 512) % 512);
                String language = LOCALES.get(name);
                if (language != null) out.addAll(parse(new ByteArrayInputStream(body), language));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (out.isEmpty()) throw new IllegalStateException("No en-US or de-DE data in " + tar);
        return out;
    }

    static List<Example<AssistantRequest>> parse(InputStream jsonl, String language) throws IOException {
        List<Example<AssistantRequest>> out = new ArrayList<>();
        BufferedReader r = new BufferedReader(new InputStreamReader(jsonl, StandardCharsets.UTF_8));
        for (String line; (line = r.readLine()) != null; ) {
            if (line.isBlank()) continue;
            JsonNode n = JSON.readTree(line);
            out.add(new Example<>(State.of(n.get("utt").asText()), new AssistantRequest(
                    ShowcaseDataset.constant(AssistantRequest.Scenario.class, n.get("scenario").asText())), language));
        }
        return out;
    }

    private static String field(byte[] header, int offset, int length) {
        int end = offset;
        while (end < offset + length && header[end] != 0) end++;
        return new String(header, offset, end - offset, StandardCharsets.US_ASCII);
    }
}
