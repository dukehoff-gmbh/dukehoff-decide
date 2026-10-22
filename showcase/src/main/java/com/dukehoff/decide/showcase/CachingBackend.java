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

import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.LabelPrompt;
import com.dukehoff.decide.spi.ScoringBackend;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalLong;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Remembers raw scores on disk so that re-running the showcase (other thresholds, other splits)
 * does not query the model again. Each field is keyed by the hash of its exact {@link LabelPrompt},
 * so changing the prompt, the schema or the ticket text is a cache miss. One file per model id.
 *
 * <p>Also records how long the uncached call for each ticket took, for the latency report.
 */
final class CachingBackend implements ScoringBackend {

    /** @param millis wall time of the backend call that produced this entry (all fields of one ticket) */
    record Entry(double[] logProbs, long millis) {}

    private static final ObjectMapper JSON = new ObjectMapper();

    private final ScoringBackend delegate;
    private final Path file;
    private final Map<String, Entry> entries = new ConcurrentHashMap<>();

    CachingBackend(ScoringBackend delegate, Path dir) {
        this.delegate = delegate;
        this.file = dir.resolve(delegate.modelId().replaceAll("[^A-Za-z0-9._-]+", "_") + ".json");
        if (Files.exists(file)) {
            try {
                entries.putAll(JSON.readValue(file.toFile(), new TypeReference<Map<String, Entry>>() {}));
            } catch (IOException e) {
                throw new UncheckedIOException("Cannot read score cache " + file, e);
            }
        }
    }

    @Override
    public String modelId() {
        return delegate.modelId();
    }

    @Override
    public Map<String, RawScores> score(ScoringRequest request) {
        Map<String, String> keys = keys(request);
        if (!keys.values().stream().allMatch(entries::containsKey)) {
            long start = System.nanoTime();
            Map<String, RawScores> scores = delegate.score(request);
            long millis = (System.nanoTime() - start) / 1_000_000;
            keys.forEach((field, key) -> entries.put(key, new Entry(scores.get(field).logProbs(), millis)));
        }
        Map<String, RawScores> out = new LinkedHashMap<>();
        keys.forEach((field, key) -> out.put(field, new RawScores(entries.get(key).logProbs())));
        return out;
    }

    /** Wall time of the model call for this request, if it has been scored. */
    OptionalLong millis(ScoringRequest request) {
        long max = 0;
        for (String key : keys(request).values()) {
            Entry e = entries.get(key);
            if (e == null) return OptionalLong.empty();
            max = Math.max(max, e.millis());
        }
        return OptionalLong.of(max);
    }

    synchronized void save() {
        try {
            Files.createDirectories(file.getParent());
            Path tmp = Files.createTempFile(file.getParent(), "scores", ".json");
            JSON.writeValue(tmp.toFile(), new TreeMap<>(entries));
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write score cache " + file, e);
        }
    }

    Path file() {
        return file;
    }

    private static Map<String, String> keys(ScoringRequest request) {
        Map<String, String> keys = new LinkedHashMap<>();
        for (FieldSpec f : request.schema().fields()) {
            LabelPrompt p = LabelPrompt.of(request.schema(), f, request.state());
            keys.put(f.name(), sha256(p.system() + '\0' + p.user()));
        }
        return keys;
    }

    private static String sha256(String s) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
