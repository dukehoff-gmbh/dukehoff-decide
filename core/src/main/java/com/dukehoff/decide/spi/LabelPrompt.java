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
package com.dukehoff.decide.spi;

import com.dukehoff.decide.DecideException;
import com.dukehoff.decide.State;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * A multiple-choice prompt for one field, shared by logprob backends.
 *
 * <p>Options are shown under the labels "A", "B", "C", ... because single capital letters are
 * one token in practically every tokenizer and never collide with each other. The backend asks
 * for exactly one token, reads the log-probabilities of the candidate tokens and hands them to
 * {@link #scores(Map, double)}, which maps them back to {@link FieldSpec#options()} order.
 *
 * @param system system message: task framing plus the schema's instructions
 * @param user   user message: the state, the question and the labelled options
 * @param labels one label per option, in option order
 */
public record LabelPrompt(String system, String user, List<String> labels) {

    /** One per letter of the alphabet. */
    public static final int MAX_OPTIONS = 26;

    public LabelPrompt {
        labels = List.copyOf(labels);
    }

    public static LabelPrompt of(TaskSchema schema, FieldSpec field, State state) {
        List<FieldSpec.OptionSpec> options = field.options();
        if (options.size() > MAX_OPTIONS) {
            throw new DecideException("Field '" + field.name() + "' of " + schema.taskId() + " has "
                    + options.size() + " options; label prompts support at most " + MAX_OPTIONS);
        }
        StringBuilder system = new StringBuilder(
                "You answer multiple-choice questions about an input. "
                        + "Reply with the letter of exactly one option and nothing else.");
        if (!schema.instructions().isBlank()) {
            system.append("\n\n").append(schema.instructions().strip());
        }

        List<String> labels = new ArrayList<>();
        StringBuilder user = new StringBuilder();
        user.append("Input:\n").append(state.render()).append("\n\n");
        user.append("Question: ").append(field.question().strip()).append("\n\nOptions:\n");
        for (int i = 0; i < options.size(); i++) {
            String label = String.valueOf((char) ('A' + i));
            labels.add(label);
            user.append(label).append(") ").append(render(field, options.get(i))).append('\n');
        }
        user.append("\nAnswer with a single letter.");
        return new LabelPrompt(system.toString(), user.toString(), labels);
    }

    /**
     * SHA-256 (hex) of everything that shapes this field's prompt except the input: the template, the
     * instructions, the question, and the options with their labels, descriptions and order. Calibration
     * profiles store it, so that editing an {@code @Question} or {@code @Option} text, reordering enum
     * constants or changing this template is noticed instead of silently reusing a profile fitted for other
     * prompts.
     */
    public static String fingerprint(TaskSchema schema, FieldSpec field) {
        LabelPrompt p = of(schema, field, State.of("\u0000input\u0000"));
        StringBuilder sb = new StringBuilder(p.system()).append('\u0000').append(p.user());
        for (FieldSpec.OptionSpec o : field.options()) sb.append('\u0000').append(o.label());
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(sb.toString().getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static String render(FieldSpec field, FieldSpec.OptionSpec option) {
        // scale options are labelled "1".."n" internally; the description carries the value and anchor
        if (field instanceof FieldSpec.ScoreSpec) return option.description();
        return option.description().isBlank() ? option.label() : option.label() + ": " + option.description();
    }

    /**
     * Maps candidate next tokens to scores in option order. Tokens are normalised first
     * (whitespace and punctuation such as {@code " A"}, {@code "A)"} or {@code "**A"} are stripped),
     * and variants of the same label are merged with log-sum-exp.
     *
     * @param tokenLogProbs candidate tokens and their log-probabilities, e.g. an API's top-k list
     * @param missing       score for labels that do not appear among the candidates
     */
    public double[] scores(Map<String, Double> tokenLogProbs, double missing) {
        double[] out = new double[labels.size()];
        java.util.Arrays.fill(out, Double.NEGATIVE_INFINITY);
        tokenLogProbs.forEach((token, logProb) -> {
            int i = labels.indexOf(normalise(token));
            if (i >= 0) out[i] = logSumExp(out[i], logProb);
        });
        for (int i = 0; i < out.length; i++) {
            if (out[i] == Double.NEGATIVE_INFINITY) out[i] = missing;
        }
        return out;
    }

    static String normalise(String token) {
        int start = 0, end = token.length();
        while (start < end && !Character.isLetterOrDigit(token.charAt(start))) start++;
        while (end > start && !Character.isLetterOrDigit(token.charAt(end - 1))) end--;
        return token.substring(start, end);
    }

    private static double logSumExp(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }
}
