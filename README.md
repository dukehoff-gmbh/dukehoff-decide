<p align="center">
  <img src="docs/dukehoff-decide-logo.svg" alt="dukehoff-decide logo: Duke holding a checklist and pointing at a ticked option" width="180">
</p>

# dukehoff-decide

Type-safe, calibrated decisions from small language models, for Java.

Many LLM integrations ask a large model a question whose answer is really one of a few options:
"which team handles this ticket", "is this spam", "how relevant is this document". dukehoff-decide turns
such a question into a Java type and answers it using a small (local) model. Instead of text, you
get a typed value plus probabilities you can rely on, and a gate that sends uncertain cases to a human.

```java
@Question("Triage an incoming customer support ticket")
record Triage(
    @Question("How urgent is this ticket?") Priority priority,
    @Question("Which team should handle it?") Team team,
    @Question("Does the customer threaten to cancel?") boolean churnRisk,
    @Question("How angry is the customer?")
    @Scale(min = 0, max = 1, steps = 5, anchors = {"calm", "furious"}) double anger) {}

DecisionTask<Triage> triage = decider.task(Triage.class);

Decision<Triage> d = triage.decide(State.of(ticketText));
d.value();                      // always a valid Triage
d.distribution("priority");     // Map<Priority, Double>

switch (triage.requireConfidence(Map.of("priority", 0.7, "churnRisk", 0.9)).decide(state)) {
    case Outcome.Confident<Triage>(var dec)          -> route(dec.value());
    case Outcome.Uncertain<Triage>(var dec, var why) -> sendToHuman(state, dec, why);
}
```

## How it works

1. **Schema.** The output type is reflected into questions. Every field becomes "pick one of N options":
   a boolean is yes/no, an enum has one option per constant (described with `@Option`), and a `@Scale`
   has one option per step.
2. **Backend.** A `ScoringBackend` returns raw log-probabilities per option. The included `OpenAiBackend`
   asks an OpenAI-compatible server for exactly one token and reads the log-probabilities of the option
   labels `A`, `B`, `C`, ...
3. **Calibration.** Raw model probabilities are usually overconfident, often badly, and small models often
   favour particular options. `Calibration.fit` fits a temperature per field from a few hundred labelled
   examples, and additionally a bias per option where that does better on a holdout set. It reports accuracy
   and expected calibration error (ECE) before and after, shows which options get confused, and produces a
   profile per (task, model).
4. **Binding.** Calibrated probabilities become a typed value, per-field distributions, and an optional gate
   that separates confident decisions from ones that need a human.

## Getting started

Requires Java 21 and a server that exposes `/v1/chat/completions` with `logprobs`. Verified with
llama.cpp (`llama-server`) and Ollama. The quickest start is Ollama:

```
brew install ollama        # or see ollama.com
ollama serve &
ollama pull qwen2.5:1.5b
```

```xml
<dependency>
    <groupId>com.dukehoff.decide</groupId>
    <artifactId>dukehoff-decide-openai</artifactId>   <!-- brings dukehoff-decide-core -->
    <version>0.1.0</version>
</dependency>
```

```java
ScoringBackend backend = OpenAiBackend.builder()
        .baseUrl("http://localhost:11434/v1")      // Ollama; llama-server defaults to :8080/v1
        .model("qwen2.5:1.5b")
        .build();
CalibrationStore profiles = CalibrationStore.directory(Path.of("src/main/resources/decide/calibration"));
```

**1. Calibrate once, offline.** Use a few hundred historical cases with known answers, for example tickets
and the team that finally handled them:

```java
DecisionTask<Triage> raw = Decider.builder().backend(backend).build().task(Triage.class);
var report = Calibration.fit(raw, train, holdout);   // List<Calibration.Labeled<Triage>> each
System.out.println(report);   // accuracy, ECE, coverage per threshold, top confusions, warnings
profiles.save(report.profile());

double threshold = report.thresholdFor(Calibration.WHOLE_RECORD, 0.95).orElse(1.0);
```

The report shows, per field and for the whole record, how many cases a gate at each threshold would decide
automatically and how accurate those are. `thresholdFor` picks the lowest threshold that reached a target
accuracy on the holdout set; treat it as an estimate and re-check it as new labelled cases come in.

The report says per field whether temperature scaling or temperature plus per-option bias was chosen, and
why (holdout NLL of both). A bias can change which option wins, which is the point: it corrects a model that
systematically over-predicts an option. Read the report. A warning like `category: option CONTACT receives 16 of 30 errors (53%); its description
probably matches too much` means an `@Option` description needs sharpening. In the showcase, rewording one
description raised accuracy from 76% to 88–90%, more than a 4.7× larger model did.

**2. Decide at runtime** with the stored profile:

```java
Decider decider = Decider.builder()
        .backend(backend)
        .calibration(CalibrationStore.classpath("decide/calibration"))
        .requireCalibration(true)       // fail fast if the profile is missing
        .build();
DecisionTask<Triage> triage = decider.task(Triage.class);
```

Profiles are keyed by output type and model id (by default the model name), so switching the model means
calibrating again. Each profile also records a fingerprint of every field's prompt, so editing an `@Question` or
`@Option` text or reordering enum constants is noticed: the decider logs a warning, or refuses to start with
`requireCalibration(true)`. Switching the runtime counts as a new model too: Ollama's `qwen2.5:1.5b` and Qwen's own
`Q4_K_M` GGUF for llama.cpp are both 4-bit Qwen2.5 1.5B, but different files, and in an earlier version of the
showcase they reached 82.0% and 87.7% category accuracy on the same messages. Give each setup its own `modelId`.

## Results

From `dukehoff-decide-showcase` (2026-10-25), Qwen2.5-Instruct (Q4_K_M) via a native llama.cpp server on an
M1 Pro, 300 calibration and 300 disjoint test examples per run (seed 42). For comparison, a linear baseline without
any language model (`-Dshowcase.backend=linear`): TF-IDF features and logistic regression per field, trained on
examples that occur in neither split.

**Customer messages** ([Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
11 categories plus an "offensive language" flag), labels follow from the text. Temperature only → with per-option
bias (the default):

| | 1.5B | 7B | linear, 20,000 training examples | most frequent label |
|---|---:|---:|---:|---:|
| category accuracy | 83.3% → 87.0% | 87.7% → 91.3% | 100% | 22.0% |
| whole record decided at threshold 0.9 | 61% → 66% | 68% → 78% | 100% | |
| accuracy of those | 90.7% → 99.5% | 97.5% → 99.1% | 100% | |
| time per message | 0.22 s | 0.98 s | 27 µs | |

Both models over-predict `CONTACT` ("reach customer service"); the fitted bias pushes it down (−2.1 and −1.7),
which raises its precision from 35% to 53% (1.5B) and from 56% to 78% (7B) without losing any of its cases. The
bias is fitted on the calibration split and only kept where it beats temperature scaling on the test split, so
the test numbers are slightly optimistic for the chosen calibrator. Time per message is wall time over the whole
run, four messages in flight.

The linear baseline needs labelled data instead of a prompt: 86% category accuracy with 100 training examples,
98.3% with 300, 100% from 3,000. That is not memorised near-copies: with every training message dropped whose
character-trigram similarity (Jaccard) to a calibration or test message is 0.4 or more (`-Dshowcase.similarity=0.4`,
three quarters of the candidates), it still reaches 99% from 1,000 examples. Bitext categories are named by a few
words ("invoice", "refund", "shipping address"), which is easy for word counts.

**Voice assistant requests** ([Amazon MASSIVE 1.1](https://github.com/alexa/massive), CC BY 4.0, crowd-written,
18 scenarios such as alarm, calendar, music, weather; German is translated from English, so each language runs on
its own). Harder than Bitext: the scenario is often implied, not named. Temperature only → with per-option bias:

| | 1.5B | 7B | linear, ~15,900 training examples | most frequent label |
|---|---:|---:|---:|---:|
| accuracy, English | 64.3% → 73.3% | 77.3% → 83.0% | 88.3% | 12.0% |
| accuracy, German | 59.7% → 68.0% | 74.3% → 78.3% | 85.0% | 12.0% |
| decided at threshold 0.9, English | 31% → 43% | 37% → 58% | 80% | |
| accuracy of those | 84.8% → 86.7% | 96.4% → 93.1% | 95.9% | |
| decided at threshold 0.9, German | 22% → 30% | 28% → 50% | 73% | |
| accuracy of those | 82.1% → 91.2% | 94.0% → 89.4% | 96.3% | |

Here the number of labels decides. The linear model's accuracy by number of training examples (English / German):
51% / 51% with 100, 66% / 67% with 300, 78% / 76% with 1,000, 82% / 80% with 3,000, 88% / 87% with 10,000. With
the 300 labelled examples that calibration needs anyway, the language models are clearly better; the linear model
catches up with the 1.5B model at about 1,000 examples and with the 7B model at about 3,000. The bias mostly
pushes the models away from `GENERAL` (small talk) and towards `PLAY`; `QA` vs. `GENERAL` is the most common
confusion. With bias, the 7B model decides more requests at threshold 0.9 but those are slightly less accurate,
so check `thresholdFor` rather than reusing a threshold.

**Support tickets** ([Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
synthetic, type/queue/priority), where many labels cannot be inferred from the text. Accuracy after calibration:

| | 1.5B | 7B | linear, 19,398 training examples | most frequent label |
|---|---:|---:|---:|---:|
| type | 65.3% | 60.7% | 83.7% | 38.3% |
| queue | 33.0% | 26.3% | 54.0% | 26.3% |
| priority | 39.7% | 38.3% | 59.3% | 39.0% |

The language models barely beat always guessing the most frequent label on `queue` and `priority`, yet often
claim over 99% confidence: raw scores would auto-route 42% (1.5B) to 65% (7B) of tickets at threshold 0.9, with
under 10% of them fully correct. After calibration, none are auto-routed, which is the right answer. The
calibration report flags `queue` and `priority` ("temperature at the limit") and the options that collect most
errors. The linear model learns how this dataset assigns queues and priorities, which no prompt describes, but
it is not good enough to auto-route whole tickets either.

## Limitations

- At most 26 options per field (labels `A`–`Z`).
- One model request per field; a decision with three fields makes three requests (they run concurrently).
- Calibration is a temperature plus one bias per option. It corrects a model that over- or under-predicts an
  option across the board, not confusions that depend on the input.
- `OpenAiBackend` has been verified against llama.cpp and Ollama (0.34). Other OpenAI-compatible servers
  should work if they return `top_logprobs`; OpenAI's reasoning models do not.
- Ollama handles one request at a time by default. Set `OLLAMA_NUM_PARALLEL` on the server if requests
  should run concurrently.

## Modules

| module | |
|---|---|
| `dukehoff-decide-core` | API, schema binding, gating, calibration, diagnostics, profile stores. No runtime dependencies. |
| `dukehoff-decide-openai` | Backend for OpenAI-compatible `/v1/chat/completions` with logprobs. Uses `java.net.http` and Jackson. |
| `dukehoff-decide-spring-boot-starter` | Spring Boot 4 auto-configuration, `DecisionTask<T>` beans, health indicator, Micrometer metrics. |
| `dukehoff-decide-showcase` | End-to-end evaluation on public datasets (not published). |

## Spring Boot

```xml
<dependency>
    <groupId>com.dukehoff.decide</groupId>
    <artifactId>dukehoff-decide-spring-boot-starter</artifactId>
    <version>0.1.0</version>
</dependency>
```

```yaml
dukehoff:
  decide:
    openai:
      base-url: http://localhost:8080/v1   # any OpenAI-compatible server with logprobs
      model: qwen2.5-7b-instruct-q4_k_m
    calibration:
      location: classpath:decide/calibration   # profiles fitted offline, shipped with the app
      required: true                           # fail at startup if a profile is missing
```

```java
@SpringBootApplication
@EnableDecisionTasks({Triage.class})
class App {}

@Service
class TicketService {
    private final DecisionTask.Gated<Triage> triage;

    TicketService(DecisionTask<Triage> triage) {
        this.triage = triage.requireConfidence(0.9);
    }
}
```

The starter creates an `OpenAiBackend` when `dukehoff.decide.openai.model` is set, a `CalibrationStore` from
`dukehoff.decide.calibration.location` (`classpath:...`, `file:...` or a path), a `Decider`, and one `DecisionTask<T>` bean
per type in `@EnableDecisionTasks`. Your own `ScoringBackend`, `CalibrationStore` or `Decider` bean replaces the
default. Tasks are created at startup, so an invalid output type or a missing required profile stops the
application immediately. Environment variables work as usual, e.g. `DUKEHOFF_DECIDE_OPENAI_MODEL`.

With Actuator, the health component `dukehoffDecide` checks that the model server answers `GET /models`
(switch off: `management.health.dukehoff-decide.enabled=false`). With Micrometer, every task records, tagged `task` and `model`:

| meter | type | extra tags |
|---|---|---|
| `dukehoff.decide.decisions` | timer | `mode` (single, batch), `outcome` (success, error) |
| `dukehoff.decide.decision.confidence` | distribution summary | |
| `dukehoff.decide.gated` | counter | `outcome` (confident, uncertain); uncertain / total is the deferral rate |

Switch off with `dukehoff.decide.metrics.enabled=false`. Metrics apply to your own `Decider` bean too.

## Running the showcase

Start a local model server, for example llama.cpp in Docker (CPU only on macOS; for realistic speed use a
native `llama-server`, e.g. `brew install llama.cpp`):

```
docker run -d --name dukehoff-decide-llm -p 8080:8080 -v ~/.cache/llama.cpp:/models -e LLAMA_CACHE=/models \
  ghcr.io/ggml-org/llama.cpp:server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 -c 4096 --cache-ram 1024
```

Then:

```
mvn -q install -DskipTests
mvn -q -pl showcase exec:java -Dshowcase.dataset=bitext -Dshowcase.model=qwen2.5-1.5b-instruct-q4_k_m
```

The report goes to `target/showcase/<dataset>/report.md` (relative to where Maven runs). Raw scores are cached in
`~/.cache/dukehoff-decide/scores`, so re-runs only query the model for new examples. On a laptop, keep the machine awake
during long runs (`caffeinate -i mvn ...` on macOS); requests in flight when it sleeps time out.
Options: `-Dshowcase.dataset=tickets|bitext|massive` (massive needs `-Dshowcase.language=en` or `de`), `-Dshowcase.calibration=300`, `-Dshowcase.test=300`,
`-Dshowcase.language=all|en|de`, `-Dshowcase.seed=42`, `-Dshowcase.baseUrl=...`, `-Dshowcase.apiKey=...`.
`-Dshowcase.backend=linear` (with `-Dshowcase.train=3000`) needs no model: it trains a TF-IDF + logistic
regression baseline on examples outside both splits and writes the same report to `target/showcase/<dataset>-linear`;
`-Dshowcase.similarity=0.6` also drops training examples that resemble a calibration or test example.
Datasets are downloaded on first run and keep their own licenses (tickets: CC BY-NC 4.0, Bitext: CDLA-Sharing-1.0,
MASSIVE: CC BY 4.0).

## Build

```
mvn verify
```

Requires Java 21. Tests need no model; the live tests run only when `DUKEHOFF_DECIDE_OPENAI_BASE_URL` and
`DUKEHOFF_DECIDE_OPENAI_MODEL` are set. Every Java file needs the license header: `mvn license:format` adds it.

Releasing: see [RELEASING.md](RELEASING.md).

## License

[Apache License 2.0](LICENSE)
