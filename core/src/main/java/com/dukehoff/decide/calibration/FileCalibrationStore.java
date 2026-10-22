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
package com.dukehoff.decide.calibration;

import com.dukehoff.decide.DecideException;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

/**
 * Stores profiles as human-readable properties files, e.g.
 * <pre>
 * taskId=com.acme.Triage
 * modelId=ollama:qwen3.5:4b
 * field.priority.temperature=1.84
 * field.churnRisk.temperature=2.31
 * field.team.temperature=1.62
 * field.team.bias.BILLING=0.41
 * field.team.bias.PLATFORM=-0.27
 * field.team.promptHash=3f9c...
 * </pre>
 * A field with {@code bias.*} entries uses {@link Calibrator.Bias}, keyed by option label so that reordering
 * enum constants is safe; otherwise {@link Calibrator.Temperature}. {@code promptHash} is the fingerprint of the
 * field's prompt at fitting time (see {@link Calibration.Profile}).
 * Plain text on purpose: profiles belong in version control next to the code that uses them.
 */
final class FileCalibrationStore implements CalibrationStore {

    private final Path dir;

    FileCalibrationStore(Path dir) {
        this.dir = dir;
    }

    static String fileName(String taskId, String modelId) {
        return (taskId + "__" + modelId).replaceAll("[^A-Za-z0-9._-]", "_") + ".properties";
    }

    /**
     * @return the profile, or empty if the file belongs to another (taskId, modelId) whose
     * sanitised file name happens to collide
     */
    static Optional<Calibration.Profile> read(Reader in, String taskId, String modelId) throws IOException {
        Properties p = new Properties();
        p.load(in);
        if (!taskId.equals(p.getProperty("taskId")) || !modelId.equals(p.getProperty("modelId"))) {
            return Optional.empty();
        }
        Map<String, Double> temperatures = new LinkedHashMap<>();
        Map<String, String> promptHashes = new LinkedHashMap<>();
        Map<String, Map<String, Double>> biases = new LinkedHashMap<>();
        for (String key : p.stringPropertyNames().stream().sorted().toList()) {
            if (!key.startsWith("field.")) continue;
            String[] parts = key.substring("field.".length()).split("\\.", 3);
            String value = p.getProperty(key).strip();
            if (parts.length == 2 && parts[1].equals("temperature")) {
                temperatures.put(parts[0], Double.parseDouble(value));
            } else if (parts.length == 2 && parts[1].equals("promptHash")) {
                promptHashes.put(parts[0], value);
            } else if (parts.length == 3 && parts[1].equals("bias")) {
                biases.computeIfAbsent(parts[0], f -> new LinkedHashMap<>()).put(parts[2], Double.parseDouble(value));
            }
        }
        Map<String, Calibrator> byField = new LinkedHashMap<>();
        temperatures.forEach((field, t) -> byField.put(field, biases.containsKey(field)
                ? new Calibrator.Bias(t, biases.get(field)) : Calibrator.temperature(t)));
        biases.forEach((field, b) -> byField.putIfAbsent(field, new Calibrator.Bias(1.0, b)));
        return Optional.of(new Calibration.Profile(taskId, modelId, byField, promptHashes));
    }

    @Override
    public Optional<Calibration.Profile> load(String taskId, String modelId) {
        Path file = dir.resolve(fileName(taskId, modelId));
        if (!Files.exists(file)) return Optional.empty();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            return read(r, taskId, modelId);
        } catch (IOException e) {
            throw new DecideException("Cannot read calibration profile " + file, e);
        }
    }

    @Override
    public void save(Calibration.Profile profile) {
        Properties p = new Properties();
        p.setProperty("taskId", profile.taskId());
        p.setProperty("modelId", profile.modelId());
        profile.promptHashByField().forEach((field, hash) -> p.setProperty("field." + field + ".promptHash", hash));
        profile.byField().forEach((field, cal) -> {
            switch (cal) {
                case Calibrator.Temperature t -> p.setProperty("field." + field + ".temperature", Double.toString(t.t()));
                case Calibrator.Identity i -> p.setProperty("field." + field + ".temperature", "1.0");
                case Calibrator.Bias b -> {
                    p.setProperty("field." + field + ".temperature", Double.toString(b.t()));
                    b.biasByLabel().forEach((label, v) ->
                            p.setProperty("field." + field + ".bias." + label, Double.toString(v)));
                }
            }
        });
        try {
            Files.createDirectories(dir);
            try (Writer w = Files.newBufferedWriter(dir.resolve(fileName(profile.taskId(), profile.modelId())),
                    StandardCharsets.UTF_8)) {
                p.store(w, "dukehoff-decide calibration profile");
            }
        } catch (IOException e) {
            throw new DecideException("Cannot write calibration profile to " + dir, e);
        }
    }
}
