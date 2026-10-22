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
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

/**
 * Read-only store for profiles packaged with the application, in the same format and with the
 * same file names as {@link FileCalibrationStore}. Fit and save with a directory store during
 * development, then ship the files as resources.
 */
final class ClasspathCalibrationStore implements CalibrationStore {

    private final String prefix;
    private final ClassLoader loader;

    ClasspathCalibrationStore(String dir, ClassLoader loader) {
        String d = dir.startsWith("/") ? dir.substring(1) : dir;
        this.prefix = d.isEmpty() || d.endsWith("/") ? d : d + "/";
        this.loader = loader;
    }

    @Override
    public Optional<Calibration.Profile> load(String taskId, String modelId) {
        String name = prefix + FileCalibrationStore.fileName(taskId, modelId);
        InputStream in = loader.getResourceAsStream(name);
        if (in == null) return Optional.empty();
        try (Reader r = new InputStreamReader(in, StandardCharsets.UTF_8)) {
            return FileCalibrationStore.read(r, taskId, modelId);
        } catch (IOException e) {
            throw new DecideException("Cannot read calibration profile classpath:" + name, e);
        }
    }

    @Override
    public void save(Calibration.Profile profile) {
        throw new UnsupportedOperationException("classpath:" + prefix + " is read-only; save profiles with "
                + "CalibrationStore.directory(...) and package them as resources");
    }
}
