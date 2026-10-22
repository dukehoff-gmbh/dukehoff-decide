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

import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Where calibration profiles live. */
public interface CalibrationStore {

    Optional<Calibration.Profile> load(String taskId, String modelId);

    void save(Calibration.Profile profile);

    /** One {@code .properties} file per (task, model) in the given directory. */
    static CalibrationStore directory(Path dir) {
        return new FileCalibrationStore(dir);
    }

    /**
     * Read-only: profiles packaged as resources under {@code dir} (e.g. {@code "decide/calibration"}),
     * in the format {@link #directory(Path)} writes. {@link #save} throws.
     */
    static CalibrationStore classpath(String dir, ClassLoader loader) {
        return new ClasspathCalibrationStore(dir, loader);
    }

    /** {@link #classpath(String, ClassLoader)} with the context class loader. */
    static CalibrationStore classpath(String dir) {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        return classpath(dir, cl != null ? cl : CalibrationStore.class.getClassLoader());
    }

    /** For tests and for fitting and using a profile in the same process. */
    static CalibrationStore inMemory() {
        Map<String, Calibration.Profile> profiles = new ConcurrentHashMap<>();
        return new CalibrationStore() {
            public Optional<Calibration.Profile> load(String taskId, String modelId) {
                return Optional.ofNullable(profiles.get(taskId + '\0' + modelId));
            }

            public void save(Calibration.Profile p) {
                profiles.put(p.taskId() + '\0' + p.modelId(), p);
            }
        };
    }
}
