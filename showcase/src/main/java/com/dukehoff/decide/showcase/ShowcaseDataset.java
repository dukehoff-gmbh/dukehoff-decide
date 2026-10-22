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
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
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
            default -> throw new IllegalArgumentException("Unknown dataset '" + id + "', use tickets or bitext");
        };
    }

    /** Deterministic shuffle, then the first {@code n} examples (optionally of one language). */
    static <T> List<Example<T>> sample(List<Example<T>> all, int n, long seed, String language) {
        List<Example<T>> pool = new ArrayList<>(language == null || language.equals("all") ? all
                : all.stream().filter(e -> language.equals(e.language())).toList());
        Collections.shuffle(pool, new Random(seed));
        if (pool.size() < n) throw new IllegalArgumentException("only " + pool.size() + " examples available");
        return List.copyOf(pool.subList(0, n));
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
