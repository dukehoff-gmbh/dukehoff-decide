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
package com.dukehoff.decide.modeljars;

import com.integrallis.models.api.Tokenizer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelsBackendTest {

    @Test
    void everyTokenThatReadsAsALabelCounts() {
        List<String> vocab = List.of("A", " A", "A)", "**B", "AB", "a", " C", "Answer", "<|im_end|>");
        Tokenizer tokenizer = new Tokenizer() {
            public int[] encode(String text) { throw new UnsupportedOperationException(); }
            public String decode(int[] tokens) { throw new UnsupportedOperationException(); }
            public String decode(int token) { return vocab.get(token); }
            public int vocabSize() { return vocab.size(); }
            public int bosToken() { return -1; }
            public int eosToken() { return 8; }
        };
        int[][] labels = ModelsBackend.labelTokens(tokenizer);
        assertArrayEquals(new int[] {0, 1, 2}, labels[0]);
        assertArrayEquals(new int[] {3}, labels[1]);
        assertArrayEquals(new int[] {6}, labels[2]);
        assertArrayEquals(new int[0], labels[3]);
    }

    @Test
    void sharedPrefixCountsLeadingEqualTokens() {
        assertEquals(2, ModelsBackend.sharedPrefix(new int[] {1, 2, 3}, new int[] {1, 2, 4, 5}));
        assertEquals(0, ModelsBackend.sharedPrefix(new int[0], new int[] {1}));
        assertEquals(3, ModelsBackend.sharedPrefix(new int[] {1, 2, 3}, new int[] {1, 2, 3}));
    }
}
