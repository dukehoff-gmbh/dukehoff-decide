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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Minimal RFC 4180 reader: quoted fields, embedded newlines and {@code ""} escapes. First row is the header. */
final class Csv {

    private Csv() {}

    static List<Map<String, String>> read(String csv) {
        List<List<String>> records = records(csv);
        if (records.isEmpty()) return List.of();
        List<String> header = records.getFirst();
        List<Map<String, String>> out = new ArrayList<>(records.size() - 1);
        for (List<String> rec : records.subList(1, records.size())) {
            Map<String, String> row = new LinkedHashMap<>();
            for (int i = 0; i < header.size(); i++) row.put(header.get(i), i < rec.size() ? rec.get(i) : "");
            out.add(row);
        }
        return out;
    }

    static List<List<String>> records(String s) {
        List<List<String>> records = new ArrayList<>();
        List<String> record = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        int i = 0, n = s.length();
        while (i < n) {
            char c = s.charAt(i++);
            if (quoted) {
                if (c != '"') field.append(c);
                else if (i < n && s.charAt(i) == '"') { field.append('"'); i++; }
                else quoted = false;
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                record.add(field.toString());
                field.setLength(0);
            } else if (c == '\n' || c == '\r') {
                if (c == '\r' && i < n && s.charAt(i) == '\n') i++;
                record.add(field.toString());
                field.setLength(0);
                records.add(record);
                record = new ArrayList<>();
            } else {
                field.append(c);
            }
        }
        if (!field.isEmpty() || !record.isEmpty()) {
            record.add(field.toString());
            records.add(record);
        }
        return records;
    }
}
