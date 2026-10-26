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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Remembers every vector of an {@link Embedder} on disk (one append-only file per model, keyed by the SHA-256 of the
 * text), so re-runs and training-size curves don't call the server again.
 */
final class CachingEmbedder implements Embedder {

    private static final int CHUNK = 1600;

    private final Embedder delegate;
    private final Path file;
    private final Map<String, float[]> cache = new ConcurrentHashMap<>();

    CachingEmbedder(Embedder delegate, Path dir) {
        this.delegate = delegate;
        this.file = dir.resolve(delegate.modelId().replaceAll("[^A-Za-z0-9._-]+", "_") + ".bin");
        load();
    }

    Embedder delegate() {
        return delegate;
    }

    @Override
    public String modelId() {
        return delegate.modelId();
    }

    /** Fetches every text not cached yet, in chunks so that progress is visible and saved. */
    @Override
    public List<float[]> embed(List<String> texts) {
        List<String> todo = texts.stream().distinct().filter(t -> !cache.containsKey(key(t))).toList();
        if (todo.size() > CHUNK) {
            System.err.printf(Locale.ROOT, "Embedding %d texts with %s, cache: %s%n", todo.size(), modelId(), file);
        }
        long start = System.nanoTime();
        for (int i = 0; i < todo.size(); i += CHUNK) {
            List<String> chunk = todo.subList(i, Math.min(todo.size(), i + CHUNK));
            List<float[]> vectors = delegate.embed(chunk);
            Map<String, float[]> fresh = new LinkedHashMap<>();
            for (int j = 0; j < chunk.size(); j++) fresh.put(key(chunk.get(j)), vectors.get(j));
            append(fresh);
            if (todo.size() > CHUNK) {
                int n = i + chunk.size();
                System.err.printf(Locale.ROOT, "  %d/%d texts, %.0f/s%n", n, todo.size(), n / ((System.nanoTime() - start) / 1e9));
            }
        }
        return texts.stream().map(t -> cache.get(key(t))).toList();
    }

    /** Wall time in ms of one uncached request per text (bypasses and does not fill the cache). */
    double[] timeSingle(List<String> texts) {
        double[] ms = new double[texts.size()];
        for (int i = 0; i < texts.size(); i++) {
            long t0 = System.nanoTime();
            delegate.embed(texts.get(i));
            ms[i] = (System.nanoTime() - t0) / 1e6;
        }
        return ms;
    }

    private void load() {
        if (!Files.exists(file)) return;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            byte[] hash = new byte[32];
            while (true) {
                try {
                    in.readFully(hash);
                } catch (EOFException end) {
                    return;
                }
                float[] v = new float[in.readInt()];
                for (int i = 0; i < v.length; i++) v[i] = in.readFloat();
                cache.put(HexFormat.of().formatHex(hash), v);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read embedding cache " + file, e);
        }
    }

    private synchronized void append(Map<String, float[]> fresh) {
        try {
            Files.createDirectories(file.getParent());
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)))) {
                for (Map.Entry<String, float[]> e : fresh.entrySet()) {
                    out.write(HexFormat.of().parseHex(e.getKey()));
                    out.writeInt(e.getValue().length);
                    for (float f : e.getValue()) out.writeFloat(f);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write embedding cache " + file, e);
        }
        cache.putAll(fresh);
    }

    private static String key(String text) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(text.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
