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
`Q4_K_M` GGUF for llama.cpp are both 4-bit Qwen2.5 1.5B, but different files, and in the showcase they
reached 82.0% and 87.7% category accuracy on the same messages. Give each setup its own `modelId`.

## Results

From `dukehoff-decide-showcase`, Qwen2.5-Instruct (Q4_K_M) via llama.cpp, 300 calibration and 300 disjoint
test examples per run.

**Customer messages** ([Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
11 categories plus an "offensive language" flag), labels follow from the text. Temperature only → with per-option
bias (the default):

| | 1.5B, llama.cpp | 7B, llama.cpp | 1.5B, Ollama | most frequent label |
|---|---:|---:|---:|---:|
| category accuracy | 87.7% → 93.7% | 90.0% → 94.0% | 82.0% → 92.7% | 21.3% |
| whole record decided at threshold 0.9 | 50% → 71% | 68% → 82% | 39% → 69% | |
| accuracy of those | 94.0% → 99.5% | 96.6% → 99.2% | 91.4% → 100% | |

All three models over-predict `CONTACT` ("reach customer service"); the fitted bias pushes it down (−1.9 to −3.0),
which doubles its precision without losing any of its cases. The bias is fitted on the calibration split and
only kept where it beats temperature scaling on the test split, so the test numbers are slightly optimistic
for the chosen calibrator. Ollama's `qwen2.5:1.5b` is a different 4-bit file of the same model; it ran at
4.4 messages per second on an M1 Pro.

**Support tickets** ([Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
synthetic, type/queue/priority), where many labels cannot be inferred from the text: only `type` beats always
guessing the most frequent label; `queue` and `priority` do not, yet the models often claim over 99% confidence.
Raw scores would
auto-route 42% (1.5B) to 65% (7B) of tickets at threshold 0.9, with under 10% of them fully correct. After
calibration, none are auto-routed, which is the right answer. The calibration report flags the priority
field ("temperature at the limit") and the options that collect most errors.

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
Options: `-Dshowcase.dataset=tickets|bitext`, `-Dshowcase.calibration=300`, `-Dshowcase.test=300`,
`-Dshowcase.language=all|en|de`, `-Dshowcase.seed=42`, `-Dshowcase.baseUrl=...`, `-Dshowcase.apiKey=...`.
Datasets are downloaded on first run and keep their own licenses (tickets: CC BY-NC 4.0, Bitext: CDLA-Sharing-1.0).

## Build

```
mvn verify
```

Requires Java 21. Tests need no model; the live tests run only when `DUKEHOFF_DECIDE_OPENAI_BASE_URL` and
`DUKEHOFF_DECIDE_OPENAI_MODEL` are set. Every Java file needs the license header: `mvn license:format` adds it.

Releasing: see [RELEASING.md](RELEASING.md).

## License

[Apache License 2.0](LICENSE)
