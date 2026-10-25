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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/** Minimal RFC 4180 reader: quoted fields, embedded newlines and {@code ""} escapes. First row is the header. */
final class Csv {

    private Csv() {}

    static List<Map<String, String>> read(String csv) {
        List<Map<String, String>> out = new ArrayList<>();
        try {
            stream(new StringReader(csv), out::add);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out;
    }

    /** Calls {@code row} for every record after the header, without holding the whole input in memory. */
    static void stream(Reader in, Consumer<Map<String, String>> row) throws IOException {
        Reader r = in instanceof BufferedReader ? in : new BufferedReader(in, 1 << 16);
        List<String> header = null;
        List<String> record = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        int c = r.read();
        while (c != -1) {
            int next = r.read();
            if (quoted) {
                if (c != '"') field.append((char) c);
                else if (next == '"') { field.append('"'); next = r.read(); }
                else quoted = false;
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                record.add(field.toString());
                field.setLength(0);
            } else if (c == '\n' || c == '\r') {
                if (c == '\r' && next == '\n') next = r.read();
                record.add(field.toString());
                field.setLength(0);
                header = emit(header, record, row);
                record = new ArrayList<>();
            } else {
                field.append((char) c);
            }
            c = next;
        }
        if (!field.isEmpty() || !record.isEmpty()) {
            record.add(field.toString());
            emit(header, record, row);
        }
    }

    private static List<String> emit(List<String> header, List<String> record, Consumer<Map<String, String>> row) {
        if (header == null) return record;
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < header.size(); i++) m.put(header.get(i), i < record.size() ? record.get(i) : "");
        row.accept(m);
        return header;
    }
}
