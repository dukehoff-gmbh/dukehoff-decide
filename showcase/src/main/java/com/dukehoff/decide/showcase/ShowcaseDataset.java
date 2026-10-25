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
import com.dukehoff.decide.calibration.Calibration.Labeled;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

/** A labelled public dataset and the typed decision it is mapped to. */
interface ShowcaseDataset<T> {

    /** Short id for {@code -Dshowcase.dataset}. */
    String id();

    /** Report heading, e.g. "support ticket triage". */
    String title();

    /** Markdown: link, license and caveats. */
    String citation();

    Class<T> type();

    /** Record components by name, for per-field accuracy. */
    Map<String, Function<T, Object>> fields();

    /** Downloads into {@code cacheDir} on first use, then parses. */
    List<Example<T>> load(Path cacheDir);

    record Example<T>(State state, T label, String language) {
        Labeled<T> labeled() {
            return new Labeled<>(state, label);
        }
    }

    static ShowcaseDataset<?> byId(String id) {
        return switch (id) {
            case "tickets" -> new TicketDataset();
            case "bitext" -> new BitextDataset();
            case "massive" -> new MassiveDataset();
            default -> throw new IllegalArgumentException("Unknown dataset '" + id + "', use tickets, bitext or massive");
        };
    }

    /** Deterministic shuffle, then the first {@code n} examples (optionally of one language). */
    static <T> List<Example<T>> sample(List<Example<T>> all, int n, long seed, String language) {
        List<Example<T>> pool = shuffled(all, seed, language);
        if (pool.size() < n) throw new IllegalArgumentException("only " + pool.size() + " examples available");
        return List.copyOf(pool.subList(0, n));
    }

    /** Training examples and how many candidates were dropped for resembling the sample. */
    record Rest<T>(List<Example<T>> examples, int dropped) {}

    /**
     * Up to {@code max} examples that {@link #sample} with the same {@code n}, seed and language did not pick,
     * skipping any whose character-trigram Jaccard similarity to an example of the sample is at least
     * {@code maxSimilarity} (1 drops only exact duplicates after normalising case and whitespace).
     */
    static <T> Rest<T> rest(List<Example<T>> all, int n, int max, long seed, String language, double maxSimilarity) {
        List<Example<T>> pool = shuffled(all, seed, language);
        List<Set<Integer>> taken = pool.subList(0, n).stream().map(e -> trigrams(e.state().render())).toList();
        Map<Integer, List<Integer>> index = new HashMap<>();
        for (int i = 0; i < taken.size(); i++) {
            for (int g : taken.get(i)) index.computeIfAbsent(g, k -> new ArrayList<>()).add(i);
        }
        List<Example<T>> out = new ArrayList<>();
        int dropped = 0;
        for (Example<T> e : pool.subList(n, pool.size())) {
            if (out.size() == max) break;
            Set<Integer> grams = trigrams(e.state().render());
            Map<Integer, Integer> shared = new HashMap<>();
            for (int g : grams) {
                for (int i : index.getOrDefault(g, List.of())) shared.merge(i, 1, Integer::sum);
            }
            boolean similar = shared.entrySet().stream().anyMatch(s -> {
                int union = grams.size() + taken.get(s.getKey()).size() - s.getValue();
                return (double) s.getValue() / union >= maxSimilarity;
            });
            if (similar) dropped++;
            else out.add(e);
        }
        return new Rest<>(List.copyOf(out), dropped);
    }

    /** Hashes of the character trigrams of the normalised text (case and whitespace runs ignored). */
    static Set<Integer> trigrams(String text) {
        String t = ' ' + text.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ") + ' ';
        Set<Integer> grams = new HashSet<>();
        for (int i = 0; i + 3 <= t.length(); i++) grams.add(t.substring(i, i + 3).hashCode());
        return grams;
    }

    private static <T> List<Example<T>> shuffled(List<Example<T>> all, long seed, String language) {
        List<Example<T>> pool = new ArrayList<>(language == null || language.equals("all") ? all
                : all.stream().filter(e -> language.equals(e.language())).toList());
        Collections.shuffle(pool, new Random(seed));
        return pool;
    }

    /** Dataset label to enum constant: "Service Outages and Maintenance" -&gt; SERVICE_OUTAGES_AND_MAINTENANCE */
    static <E extends Enum<E>> E constant(Class<E> type, String label) {
        return Enum.valueOf(type, label.strip().toUpperCase(Locale.ROOT).replaceAll("[^A-Z0-9]+", "_"));
    }

    /** Downloads {@code url} to {@code file} unless it exists. */
    static Path download(String url, Path file) {
        if (Files.exists(file)) return file;
        try {
            Files.createDirectories(file.getParent());
            Path tmp = Files.createTempFile(file.getParent(), "download", ".tmp");
            HttpResponse<Path> r;
            try (HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()) {
                r = http.send(HttpRequest.newBuilder(URI.create(url)).build(), HttpResponse.BodyHandlers.ofFile(tmp));
            }
            if (r.statusCode() != 200) {
                Files.deleteIfExists(tmp);
                throw new IOException("GET " + url + " returned " + r.statusCode());
            }
            return Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }
}
