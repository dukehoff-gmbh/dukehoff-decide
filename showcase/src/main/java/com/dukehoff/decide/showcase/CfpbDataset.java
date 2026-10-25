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
import com.dukehoff.decide.showcase.ConsumerComplaint.Product;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.zip.ZipInputStream;

/**
 * Complaint narratives that US consumers sent to the Consumer Financial Protection Bureau, from the CFPB's
 * official narratives archive (FOIA reading room, US government work, no copyright), mapped to
 * {@link ConsumerComplaint}. Uses one export (complaints received September 2023 through March 2024), so the
 * product names are consistent, and samples {@value #PER_PRODUCT} narratives per product (fixed seed), because
 * credit reporting alone is 71% of the export. Narratives are cut at {@value #MAX_CHARS} characters.
 * The label is the product the consumer chose when filing, so it can be wrong.
 */
final class CfpbDataset implements ShowcaseDataset<ConsumerComplaint> {

    static final String SOURCE = "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
            + "cfpb-consumer-complaint-database-narratives-archive/";
    static final String ZIP_URL = "https://files.consumerfinance.gov/f/documents/"
            + "CCDB_Export_5_September_2023_through_March_2024.zip";
    static final int PER_PRODUCT = 2500, MAX_CHARS = 4000;
    static final long SEED = 2023;

    static final Map<String, Product> PRODUCTS = Map.of(
            "Credit reporting or other personal consumer reports", Product.CREDIT_REPORTING,
            "Debt collection", Product.DEBT_COLLECTION,
            "Credit card", Product.CREDIT_CARD,
            "Prepaid card", Product.PREPAID_CARD,
            "Checking or savings account", Product.CHECKING_OR_SAVINGS,
            "Mortgage", Product.MORTGAGE,
            "Student loan", Product.STUDENT_LOAN,
            "Vehicle loan or lease", Product.VEHICLE_LOAN,
            "Payday loan, title loan, personal loan, or advance loan", Product.PERSONAL_LOAN,
            "Money transfer, virtual currency, or money service", Product.MONEY_TRANSFER);

    private static final ObjectMapper JSON = new ObjectMapper();

    @Override
    public String id() {
        return "cfpb";
    }

    @Override
    public String title() {
        return "consumer complaint routing";
    }

    @Override
    public String citation() {
        return "[CFPB Consumer Complaint Database narratives archive](" + SOURCE + ") (US government work, EN; "
                + "real complaints received 2023-09 to 2024-03, " + PER_PRODUCT + " per product, cut at "
                + MAX_CHARS + " characters)";
    }

    @Override
    public Class<ConsumerComplaint> type() {
        return ConsumerComplaint.class;
    }

    @Override
    public Map<String, Function<ConsumerComplaint, Object>> fields() {
        Map<String, Function<ConsumerComplaint, Object>> f = new LinkedHashMap<>();
        f.put("product", ConsumerComplaint::product);
        return f;
    }

    /** Downloads the export (100 MB) once, then caches the balanced sample as JSON lines. */
    @Override
    public List<Example<ConsumerComplaint>> load(Path cacheDir) {
        Path sample = cacheDir.resolve("cfpb-export5-" + PER_PRODUCT + "-per-product.jsonl");
        try {
            if (!Files.exists(sample)) {
                Path zip = ShowcaseDataset.download(ZIP_URL, cacheDir.resolve("CCDB_Export_5_September_2023_through_March_2024.zip"));
                writeSample(zip, sample);
            }
            List<Example<ConsumerComplaint>> out = new ArrayList<>();
            for (String line : Files.readAllLines(sample, StandardCharsets.UTF_8)) {
                JsonNode n = JSON.readTree(line);
                out.add(example(Product.valueOf(n.get("product").asText()), n.get("text").asText()));
            }
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static Example<ConsumerComplaint> example(Product product, String narrative) {
        return new Example<>(State.of(cut(narrative)), new ConsumerComplaint(product), "en");
    }

    /** At most {@value #MAX_CHARS} characters, cut at a word boundary. */
    static String cut(String text) {
        String t = text.strip();
        if (t.length() <= MAX_CHARS) return t;
        int end = t.lastIndexOf(' ', MAX_CHARS);
        return t.substring(0, end > 0 ? end : MAX_CHARS) + " …";
    }

    /** Reservoir sample of {@value #PER_PRODUCT} narratives per product, in file order of the export. */
    private static void writeSample(Path zip, Path sample) throws IOException {
        Map<Product, List<String[]>> reservoir = new EnumMap<>(Product.class);
        Map<Product, Integer> seen = new EnumMap<>(Product.class);
        Random random = new Random(SEED);
        System.err.println("Sampling " + PER_PRODUCT + " narratives per product from " + zip);
        try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
            if (in.getNextEntry() == null) throw new IOException("empty zip " + zip);
            Csv.stream(new InputStreamReader(in, StandardCharsets.UTF_8), row -> {
                Product p = PRODUCTS.get(row.get("Product"));
                String text = row.getOrDefault("Consumer complaint narrative", "");
                if (p == null || text.isBlank()) return;
                int n = seen.merge(p, 1, Integer::sum);
                List<String[]> r = reservoir.computeIfAbsent(p, k -> new ArrayList<>());
                String[] entry = {row.get("Complaint ID"), text};
                if (r.size() < PER_PRODUCT) r.add(entry);
                else {
                    int j = random.nextInt(n);
                    if (j < PER_PRODUCT) r.set(j, entry);
                }
            });
        }
        Path tmp = Files.createTempFile(sample.getParent(), "cfpb", ".jsonl");
        try (BufferedWriter w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
            for (Map.Entry<Product, List<String[]>> e : reservoir.entrySet()) {
                for (String[] entry : e.getValue()) {
                    Map<String, String> m = new LinkedHashMap<>();
                    m.put("id", entry[0]);
                    m.put("product", e.getKey().name());
                    m.put("text", entry[1]);
                    w.write(JSON.writeValueAsString(m));
                    w.newLine();
                }
            }
        }
        Files.move(tmp, sample, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        System.err.println("  per product: " + seen);
    }
}
