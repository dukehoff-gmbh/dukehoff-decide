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

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.State;
import com.dukehoff.decide.showcase.ShowcaseDataset.Example;
import com.dukehoff.decide.spi.ScoringBackend.RawScores;
import com.dukehoff.decide.spi.ScoringBackend.ScoringRequest;
import com.dukehoff.decide.testing.ScriptedBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ShowcaseTest {

    @Test
    void csvHandlesQuotesCommasAndNewlines() {
        var rows = Csv.read("subject,body,type\r\n\"Hi, there\",\"line 1\nline \"\"2\"\"\",Incident\nx,,Change");
        assertEquals(2, rows.size());
        assertEquals("Hi, there", rows.get(0).get("subject"));
        assertEquals("line 1\nline \"2\"", rows.get(0).get("body"));
        assertEquals("Incident", rows.get(0).get("type"));
        assertEquals("", rows.get(1).get("body"));
        assertEquals("Change", rows.get(1).get("type"));
    }

    @Test
    void mapsEveryDatasetLabel() {
        for (String q : List.of("Technical Support", "Product Support", "Customer Service", "IT Support",
                "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
                "Sales and Pre-Sales", "Human Resources", "General Inquiry")) {
            assertNotNull(ShowcaseDataset.constant(Triage.Queue.class, q));
        }
        assertEquals(Triage.Queue.SALES_AND_PRE_SALES, ShowcaseDataset.constant(Triage.Queue.class, "Sales and Pre-Sales"));
        assertEquals(Triage.Type.INCIDENT, ShowcaseDataset.constant(Triage.Type.class, "Incident"));
        assertEquals(Triage.Priority.MEDIUM, ShowcaseDataset.constant(Triage.Priority.class, "medium"));
    }

    @Test
    void parsesTicketsAndSkipsEmptyBodies() {
        List<Example<Triage>> t = TicketDataset.parse(List.of(
                Map.of("subject", "Down", "body", "Everything is down", "type", "Incident",
                        "queue", "Technical Support", "priority", "high", "language", "en"),
                Map.of("subject", "", "body", " ", "type", "Request",
                        "queue", "General Inquiry", "priority", "low", "language", "de")));
        assertEquals(1, t.size());
        assertEquals(new Triage(Triage.Type.INCIDENT, Triage.Queue.TECHNICAL_SUPPORT, Triage.Priority.HIGH),
                t.getFirst().label());
        assertEquals("Subject: Down\n\nEverything is down", t.getFirst().state().render());
        assertEquals("en", t.getFirst().language());
    }

    @Test
    void parsesBitextCategoryAndOffensiveFlag() {
        List<Example<CustomerRequest>> r = BitextDataset.parse(List.of(
                Map.of("flags", "BLQW", "instruction", "cancel my damn order", "category", "ORDER",
                        "intent", "cancel_order"),
                Map.of("flags", "BL", "instruction", "where is my invoice", "category", "INVOICE",
                        "intent", "check_invoice")));
        assertEquals(new CustomerRequest(CustomerRequest.Category.ORDER, true), r.get(0).label());
        assertEquals(new CustomerRequest(CustomerRequest.Category.INVOICE, false), r.get(1).label());
        assertEquals("cancel my damn order", r.get(0).state().render());
    }

    @Test
    void parsesMassiveScenarios() throws Exception {
        String jsonl = """
                {"id": "0", "locale": "de-DE", "partition": "test", "scenario": "iot", "intent": "iot_coffee", \
                "utt": "mach mir einen kaffee", "annot_utt": "mach mir einen kaffee", "worker_id": "1"}

                {"id": "1", "locale": "de-DE", "partition": "train", "scenario": "qa", "intent": "qa_factoid", \
                "utt": "wie hoch ist die zugspitze", "annot_utt": "wie hoch ist die zugspitze", "worker_id": "2"}
                """;
        List<Example<AssistantRequest>> r = MassiveDataset.parse(
                new java.io.ByteArrayInputStream(jsonl.getBytes(java.nio.charset.StandardCharsets.UTF_8)), "de");
        assertEquals(2, r.size());
        assertEquals(new AssistantRequest(AssistantRequest.Scenario.IOT), r.get(0).label());
        assertEquals(AssistantRequest.Scenario.QA, r.get(1).label().scenario());
        assertEquals("mach mir einen kaffee", r.get(0).state().render());
        assertEquals("de", r.get(0).language());
    }

    @Test
    void cutsLongComplaintsAtAWordBoundary() {
        assertEquals("short text", CfpbDataset.cut("  short text "));
        String cut = CfpbDataset.cut("word ".repeat(2000));
        assertTrue(cut.length() <= CfpbDataset.MAX_CHARS + 2, cut.length() + "");
        assertTrue(cut.endsWith("word …"));
        assertEquals(ConsumerComplaint.Product.MORTGAGE, CfpbDataset.PRODUCTS.get("Mortgage"));
    }

    @Test
    void csvStreamsLargeInputRowByRow() throws Exception {
        List<Map<String, String>> rows = new java.util.ArrayList<>();
        Csv.stream(new java.io.StringReader("a,b\r\n1,\"x\ny\"\n2,z"), rows::add);
        assertEquals(List.of(Map.of("a", "1", "b", "x\ny"), Map.of("a", "2", "b", "z")), rows);
    }

    @Test
    void everyDatasetMapsToAValidSchema() {
        for (String id : List.of("tickets", "bitext", "massive", "cfpb")) {
            ShowcaseDataset<?> ds = ShowcaseDataset.byId(id);
            var schema = Decider.builder().backend(new ScriptedBackend("s", (s, f) -> ScriptedBackend.uniform(f)))
                    .build().task(ds.type()).schema();
            assertEquals(List.copyOf(ds.fields().keySet()),
                    schema.fields().stream().map(f -> f.name()).toList(), id);
        }
    }

    @Test
    void cacheServesRepeatsAndSurvivesRestart(@TempDir Path dir) {
        var model = new ScriptedBackend("m/1:q4", (s, f) -> f.name().equals("type")
                ? new double[] {-0.1, Double.NEGATIVE_INFINITY, -3, -4}
                : ScriptedBackend.uniform(f));
        DecisionTask<Triage> task = Decider.builder().backend(model).build().task(Triage.class);
        var request = new ScoringRequest(State.of("Printer on fire"), task.schema());

        CachingBackend cache = new CachingBackend(model, dir);
        assertTrue(cache.millis(request).isEmpty());
        Map<String, RawScores> first = cache.score(request);
        cache.score(request);
        assertEquals(1, model.calls());
        cache.save();
        assertEquals("m_1_q4.json", cache.file().getFileName().toString());

        CachingBackend reloaded = new CachingBackend(model, dir);
        assertTrue(reloaded.millis(request).isPresent());
        assertArrayEquals(first.get("type").logProbs(), reloaded.score(request).get("type").logProbs());
        assertEquals(1, model.calls());

        // a different ticket is a miss
        reloaded.score(new ScoringRequest(State.of("Printer on fire!"), task.schema()));
        assertEquals(2, model.calls());
    }

    @Test
    void majorityBaselineAndPercentiles() {
        Triage a = new Triage(Triage.Type.INCIDENT, Triage.Queue.IT_SUPPORT, Triage.Priority.HIGH);
        Triage b = new Triage(Triage.Type.REQUEST, Triage.Queue.IT_SUPPORT, Triage.Priority.LOW);
        List<Example<Triage>> cal = List.of(example(a), example(a), example(b));
        List<Example<Triage>> test = List.of(example(a), example(b), example(b), example(b));
        assertEquals(0.25, Showcase.baseline(cal, test, e -> e.label().type()), 1e-12);
        assertEquals(1.0, Showcase.baseline(cal, test, e -> e.label().queue()), 1e-12);

        double[] ms = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        assertEquals(50, Showcase.percentile(ms, 0.5));
        assertEquals(100, Showcase.percentile(ms, 0.95));
    }

    private static Example<Triage> example(Triage t) {
        return new Example<>(State.of("body"), t, "en");
    }

    @Test
    void linearBackendLearnsSeparableCategoriesAndRestIsDisjoint() {
        var schema = Decider.builder().backend(new ScriptedBackend("s", (s, f) -> ScriptedBackend.uniform(f)))
                .build().task(CustomerRequest.class).schema();
        List<Example<CustomerRequest>> all = new java.util.ArrayList<>();
        for (int i = 0; i < 40; i++) {
            all.add(new Example<>(State.of("where is my invoice number " + i), new CustomerRequest(
                    CustomerRequest.Category.INVOICE, false), "en"));
            all.add(new Example<>(State.of("damn, change my shipping address to street " + i), new CustomerRequest(
                    CustomerRequest.Category.SHIPPING, true), "en"));
        }
        List<Example<CustomerRequest>> sample = ShowcaseDataset.sample(all, 10, 1, "all");
        ShowcaseDataset.Rest<CustomerRequest> rest = ShowcaseDataset.rest(all, 10, 1000, 1, "all", 1);
        List<Example<CustomerRequest>> training = rest.examples();
        assertEquals(70, training.size());
        assertEquals(0, rest.dropped());
        // "number 1" vs. "number 12" and the like: every candidate resembles some sampled message
        assertTrue(ShowcaseDataset.rest(all, 10, 1000, 1, "all", 0.5).dropped() > 0);
        assertTrue(training.stream().noneMatch(sample::contains));

        var lb = Showcase.train(new BitextDataset(), schema, training,
                new Showcase.Config("bitext", null, "linear", null, 0, 0, 1, "all", 1, null, null, null, "linear", 70, 1,
                        null, null), Showcase.linearTrainer());
        DecisionTask<CustomerRequest> task = Decider.builder().backend(lb).build().task(CustomerRequest.class);
        for (Example<CustomerRequest> e : sample) assertEquals(e.label(), task.decide(e.state()).value());
    }
}
