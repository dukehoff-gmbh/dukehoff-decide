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
 * {@code bitext/Bitext-customer-support-llm-chatbot-training-dataset} on Hugging Face (CDLA-Sharing-1.0),
 * 27k short English customer messages mapped to {@link CustomerRequest}. The category is the dataset's
 * {@code category} column; {@code offensive} is its {@code W} ("offensive language") flag.
 * Labels follow from the text, unlike {@link TicketDataset}.
 */
final class BitextDataset implements ShowcaseDataset<CustomerRequest> {

    static final String SOURCE = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset";
    static final String CSV_URL = SOURCE + "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv";

    @Override
    public String id() {
        return "bitext";
    }

    @Override
    public String title() {
        return "customer message routing";
    }

    @Override
    public String citation() {
        return "[bitext/Bitext-customer-support-llm-chatbot-training-dataset](" + SOURCE
                + ") (CDLA-Sharing-1.0, EN)";
    }

    @Override
    public Class<CustomerRequest> type() {
        return CustomerRequest.class;
    }

    @Override
    public Map<String, Function<CustomerRequest, Object>> fields() {
        Map<String, Function<CustomerRequest, Object>> f = new LinkedHashMap<>();
        f.put("category", CustomerRequest::category);
        f.put("offensive", CustomerRequest::offensive);
        return f;
    }

    @Override
    public List<Example<CustomerRequest>> load(Path cacheDir) {
        Path csv = ShowcaseDataset.download(CSV_URL, cacheDir.resolve("bitext-customer-support-27k-v11.csv"));
        try {
            return parse(Csv.read(Files.readString(csv, StandardCharsets.UTF_8)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static List<Example<CustomerRequest>> parse(List<Map<String, String>> rows) {
        List<Example<CustomerRequest>> out = new ArrayList<>(rows.size());
        for (Map<String, String> r : rows) {
            String text = r.getOrDefault("instruction", "");
            if (text.isBlank()) continue;
            out.add(new Example<>(State.of(text), new CustomerRequest(
                    ShowcaseDataset.constant(CustomerRequest.Category.class, r.get("category")),
                    r.getOrDefault("flags", "").contains("W")), "en"));
        }
        return out;
    }
}
