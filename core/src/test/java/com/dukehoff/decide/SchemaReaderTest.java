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
package com.dukehoff.decide;

import com.dukehoff.decide.Fixtures.*;
import com.dukehoff.decide.internal.SchemaReader;
import com.dukehoff.decide.spi.FieldSpec;
import com.dukehoff.decide.spi.TaskSchema;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SchemaReaderTest {

    @Test
    void recordComponentsBecomeFieldsInOrder() {
        TaskSchema s = SchemaReader.read(Triage.class, "Customer is on the enterprise plan");
        assertEquals(TaskSchema.Kind.RECORD, s.kind());
        assertEquals(Triage.class.getName(), s.taskId());
        assertEquals("Triage an incoming customer support ticket\nCustomer is on the enterprise plan", s.instructions());
        assertEquals(List.of("priority", "team", "churnRisk", "anger"),
                s.fields().stream().map(FieldSpec::name).toList());

        var priority = assertInstanceOf(FieldSpec.ChoiceSpec.class, s.field("priority"));
        assertEquals("How urgent is this ticket?", priority.question());
        assertEquals("BLOCKER", priority.options().get(0).label());
        assertEquals("Production is down or data is being lost", priority.options().get(0).description());

        var team = assertInstanceOf(FieldSpec.ChoiceSpec.class, s.field("team"));
        assertEquals("", team.options().get(0).description());

        var churn = assertInstanceOf(FieldSpec.BooleanSpec.class, s.field("churnRisk"));
        assertEquals(2, churn.options().size());
        assertEquals("yes", churn.options().get(0).label());
    }

    @Test
    void scaleIsDiscretisedIntoAnchors() {
        var anger = assertInstanceOf(FieldSpec.ScoreSpec.class,
                SchemaReader.read(Triage.class, "").field("anger"));
        assertEquals(List.of(0.0, 0.25, 0.5, 0.75, 1.0), anger.anchorValues());
        assertEquals("1", anger.options().get(0).label());
        assertEquals("0 = calm", anger.options().get(0).description());
        assertEquals("0.500", anger.options().get(2).description());
        assertEquals("1 = furious", anger.options().get(4).description());
    }

    @Test
    void enumTaskUsesInstructionsOrTypeQuestion() {
        TaskSchema fromAnnotation = SchemaReader.read(Priority.class, null);
        assertEquals(TaskSchema.Kind.ENUM, fromAnnotation.kind());
        assertEquals("How urgent is this ticket?", fromAnnotation.fields().getFirst().question());
        assertEquals(Decision.VALUE, fromAnnotation.fields().getFirst().name());

        TaskSchema fromInstructions = SchemaReader.read(Priority.class, "Urgency?");
        assertEquals("Urgency?", fromInstructions.fields().getFirst().question());

        assertThrows(DecideException.class, () -> SchemaReader.read(NoQuestion.class, ""));
    }

    @Test
    void booleanTaskNeedsInstructions() {
        TaskSchema s = SchemaReader.read(Boolean.class, "Is this spam?");
        assertEquals(TaskSchema.Kind.BOOLEAN, s.kind());
        assertEquals("Is this spam?", s.fields().getFirst().question());
        assertThrows(DecideException.class, () -> SchemaReader.read(Boolean.class, " "));
    }

    @Test
    void invalidTypesFailWithHelpfulMessages() {
        var missing = assertThrows(DecideException.class, () -> SchemaReader.read(MissingQuestion.class, ""));
        assertTrue(missing.getMessage().contains("MissingQuestion.priority needs @Question"), missing.getMessage());

        var unscaled = assertThrows(DecideException.class, () -> SchemaReader.read(UnscaledNumber.class, ""));
        assertTrue(unscaled.getMessage().contains("@Scale"), unscaled.getMessage());

        var unsupported = assertThrows(DecideException.class, () -> SchemaReader.read(UnsupportedType.class, ""));
        assertTrue(unsupported.getMessage().contains("String"), unsupported.getMessage());

        assertThrows(DecideException.class, () -> SchemaReader.read(BadAnchors.class, ""));
        assertThrows(DecideException.class, () -> SchemaReader.read(String.class, ""));
    }
}
