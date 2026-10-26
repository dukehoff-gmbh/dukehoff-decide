<p align="center">
  <img src="docs/dukehoff-decide-logo.svg" alt="dukehoff-decide logo: Duke holding a checklist and pointing at a ticked option" width="180">
</p>

# dukehoff-decide

Type-safe, calibrated decisions for Java, from small language models or from classifiers trained on your own data.

Many LLM integrations ask a large model a question whose answer is really one of a few options:
"which team handles this ticket", "is this spam", "how relevant is this document". dukehoff-decide turns
such a question into a Java type. Instead of text, you get a typed value plus probabilities you can rely on,
and a gate that sends uncertain cases to a human.

Who answers the question is a pluggable backend. With no labelled data, a small (local) language model
answers it from the question and option descriptions alone. Once you have a few thousand labelled cases, a
classifier trained on them is usually more accurate and much faster (see [Results](#results)).
Calibration, gating and diagnostics are the same either way, so you can start with a language model and
switch when the labels are there, without changing the code that uses the decisions.

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
2. **Backend.** A `ScoringBackend` returns raw log-probabilities per option; it is the only model-specific
   interface. The included `OpenAiBackend` asks an OpenAI-compatible server for exactly one token and reads
   the log-probabilities of the option labels `A`, `B`, `C`, ... A trained classifier implements the same
   interface by returning its log-softmax scores (see [Choosing a backend](#choosing-a-backend)).
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
    <version>0.2.0</version>
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

### With a trained classifier

Once there are a few thousand labelled cases, train a classifier on them instead. `dukehoff-decide-classifier`
needs no model server and has no dependencies besides `dukehoff-decide-core`:

```xml
<dependency>
    <groupId>com.dukehoff.decide</groupId>
    <artifactId>dukehoff-decide-classifier</artifactId>
    <version>0.2.0</version>
</dependency>
```

```java
LinearClassifier model = LinearClassifier.train(Triage.class, labeled);   // List<Calibration.Labeled<Triage>>
model.save(Path.of("src/main/resources/decide/triage.model"));

DecisionTask<Triage> raw = Decider.builder().backend(model).build().task(Triage.class);
profiles.save(Calibration.fit(raw, train, holdout).profile());   // cases the model was not trained on
```

`LinearClassifier` is a `ScoringBackend` like `OpenAiBackend`: TF-IDF features of `State.render()` (word 1–2-grams
and character 3–5-grams) and one logistic regression per field, trained in seconds to about a minute (20,000 texts) and scoring in well
under a millisecond. Everything above, calibration, gating, `thresholdFor` and the report, works unchanged.
Calibrate on cases the model has not seen; its scores on its own training data are too confident. At runtime, load
it with `LinearClassifier.load(path)` or `load(inputStream)` (e.g. from the classpath). Its model id is derived
from the weights, so a retrained model needs a new calibration profile. Unlike a language model it learns only
from labels, not from `@Question` or `@Option` texts: adding or reordering an option means retraining, and the
model refuses to score a task whose options no longer match. `LinearClassifier.trainer()` sets epochs, learning
rate, regularisation, the number of hash buckets and the seed.

### With embeddings

`EmbeddingClassifier` (in `dukehoff-decide-embedding`) trains the same kind of classifier on sentence embeddings
instead of word counts. An embedding model, which stays frozen, turns each text into one vector in which texts with
similar meaning lie close together, even without shared words. It needs an embedding server, e.g.
`llama-server -hf ggml-org/bge-m3-Q8_0-GGUF --embeddings --port 8081 -ub 8192 -b 8192` (the batch sizes must hold
the longest text in tokens), and pays off on longer texts written by people
and in German (see [Results](#results)).

```java
Embedder embedder = OpenAiEmbedder.builder()            // any OpenAI-compatible /v1/embeddings endpoint
        .baseUrl("http://localhost:8081/v1")
        .model("bge-m3")
        .build();
EmbeddingClassifier model = EmbeddingClassifier.train(embedder, Triage.class, labeled);
model.save(Path.of("src/main/resources/decide/triage.model"));
// at runtime
ScoringBackend backend = EmbeddingClassifier.load(Path.of(".../triage.model"), embedder);
```

The saved model holds the classifier only, not the embedding model; `load` checks that the embedder has the
model id it was trained with. Implement `Embedder` to use another runtime or to cache vectors.

## Choosing a backend

Summary of the [results](#results) below, all calibrated and gated the same way:

| | prompted language model | word counts + logistic regression (`LinearClassifier`) | sentence embeddings + logistic regression (`EmbeddingClassifier`) |
|---|---|---|---|
| labelled data | a few hundred, for calibration | a few thousand | a few thousand, fewer than word counts |
| accuracy | behind both once there are enough labels | as good as embeddings on short texts, a few points behind on long ones | best on long texts written by people, and in German |
| time per decision (M1 Pro) | 0.2–2 s | under 1 ms | 20–100 ms (one embedding) |
| needs | a model server with logprobs | nothing | an embedding server |
| changing the question | edit `@Question`/`@Option`, recalibrate | relabel and retrain | relabel and retrain |

A language model is the way to start: it works from day one, and its confident decisions are already useful
while labels accumulate (from humans handling the uncertain cases, for example). Word counts + logistic
regression is [`LinearClassifier`](#with-a-trained-classifier) in `dukehoff-decide-classifier`; sentence
embeddings + logistic regression is [`EmbeddingClassifier`](#with-embeddings) in `dukehoff-decide-embedding`.

## Results

From `dukehoff-decide-showcase` (2026-10-25), Qwen2.5-Instruct (Q4_K_M) via a native llama.cpp server on an
M1 Pro, 300 calibration and 300 disjoint test examples per run (seed 42). For comparison, a linear baseline without
any language model (`LinearClassifier`, `-Dshowcase.backend=linear`): TF-IDF features and logistic regression per field, trained on
examples that occur in neither split. And the same classifier on sentence embeddings (`-Dshowcase.backend=embedding`,
[bge-m3](https://huggingface.co/BAAI/bge-m3) or Qwen3-Embedding-0.6B via llama.cpp): a frozen pretrained encoder turns
each text into one vector that also captures context, and only the logistic regression on top is trained.

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
words ("invoice", "refund", "shipping address"), which is easy for word counts. Embeddings reach the same: 100%
(bge-m3) and 99.7% (Qwen3-Embedding).

**Voice assistant requests** ([Amazon MASSIVE 1.1](https://github.com/alexa/massive), CC BY 4.0, crowd-written,
18 scenarios such as alarm, calendar, music, weather; German is translated from English, so each language runs on
its own). Harder than Bitext: the scenario is often implied, not named. Temperature only → with per-option bias:

| | 1.5B | 7B | linear, ~15,900 training examples | bge-m3 + linear | most frequent label |
|---|---:|---:|---:|---:|---:|
| accuracy, English | 64.3% → 73.3% | 77.3% → 83.0% | 88.3% | 88.3% | 12.0% |
| accuracy, German | 59.7% → 68.0% | 74.3% → 78.3% | 85.0% | 88.0% | 12.0% |
| decided at threshold 0.9, English | 31% → 43% | 37% → 58% | 80% | 80% | |
| accuracy of those | 84.8% → 86.7% | 96.4% → 93.1% | 95.9% | 96.7% | |
| decided at threshold 0.9, German | 22% → 30% | 28% → 50% | 73% | 70% | |
| accuracy of those | 82.1% → 91.2% | 94.0% → 89.4% | 96.3% | 95.7% | |

Here the number of labels decides. The linear model's accuracy by number of training examples (English / German):
51% / 51% with 100, 66% / 67% with 300, 78% / 76% with 1,000, 82% / 80% with 3,000, 88% / 87% with 10,000. With
the 300 labelled examples that calibration needs anyway, the language models are clearly better; the linear model
catches up with the 1.5B model at about 1,000 examples and with the 7B model at about 3,000. The bias mostly
pushes the models away from `GENERAL` (small talk) and towards `PLAY`; `QA` vs. `GENERAL` is the most common
confusion. With bias, the 7B model decides more requests at threshold 0.9 but those are slightly less accurate,
so check `thresholdFor` rather than reusing a threshold. Embeddings need fewer labels than word counts (bge-m3:
67% / 62% with 100, 77% / 70% with 300, 82% / 79% with 1,000) and, being multilingual, hold up better in German.
Qwen3-Embedding gets 90.3% / 85.7%. Computing one embedding takes 16–24 ms on the M1 Pro.

**Support tickets** ([Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
synthetic, type/queue/priority), where many labels cannot be inferred from the text. Accuracy after calibration:

| | 1.5B | 7B | linear, 19,398 training examples | bge-m3 + linear | most frequent label |
|---|---:|---:|---:|---:|---:|
| type | 65.3% | 60.7% | 83.7% | 79.7% | 38.3% |
| queue | 33.0% | 26.3% | 54.0% | 44.0% | 26.3% |
| priority | 39.7% | 38.3% | 59.3% | 39.3% | 39.0% |

The language models barely beat always guessing the most frequent label on `queue` and `priority`, yet often
claim over 99% confidence: raw scores would auto-route 42% (1.5B) to 65% (7B) of tickets at threshold 0.9, with
under 10% of them fully correct. After calibration, none are auto-routed, which is the right answer. The
calibration report flags `queue` and `priority` ("temperature at the limit") and the options that collect most
errors. The linear model learns how this dataset assigns queues and priorities, which no prompt describes, but
it is not good enough to auto-route whole tickets either. Embeddings do worse than word counts here (Qwen3-Embedding:
76.3% / 42.0% / 43.3%): whatever decides queue and priority in this synthetic data is in the words, not the meaning.

**Consumer complaints** ([CFPB consumer complaint narratives](https://www.consumerfinance.gov/data-research/consumer-complaints/),
public domain, written by real consumers to the US Consumer Financial Protection Bureau, from the archived export
September 2023 to March 2024; 10 financial products, 2,500 complaints sampled per product). The longest texts
(median 117 words, cut at 4,000 characters):

| | 1.5B | 7B | linear, 20,000 training examples | bge-m3 + linear | Qwen3-Embedding + linear | most frequent label |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 64.0% → 65.7% | 70.3% → 74.3% | 78.0% | 82.3% | 80.7% | 10.7% |
| decided at threshold 0.9 | 5% | 8% → 23% | 50% | 46% | 48% | |
| accuracy of those | 71.4% → 100% | 100% → 98.5% | 97.3% | 97.1% | 96.6% | |
| time per complaint | 0.38 s | 1.8 s | 0.4 ms | 54 ms | 105 ms | |

The only dataset where embeddings beat word counts, by 2–4 points. Both need data for it: with 300 training
examples, word counts get 56% and embeddings 67–70%, below the 7B model; at 3,000, all three are at 73–74%. Every
method confuses money transfers with checking accounts most often, and many complaints are about both. Times for
the language models are wall time over the run with four complaints in flight; for embeddings, one request with
nothing else in flight.

## Limitations

- The trained classifiers need labelled data and have to be retrained when options change; they learn from
  labels only, so a reworded question changes nothing.
- With a language model: at most 26 options per field (labels `A`–`Z`), and one model request per field; a
  decision with three fields makes three requests (they run concurrently).
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
| `dukehoff-decide-classifier` | `LinearClassifier`: a backend trained on labelled examples (TF-IDF + logistic regression). No dependencies. |
| `dukehoff-decide-embedding` | `EmbeddingClassifier` (embeddings + logistic regression) and `OpenAiEmbedder` for `/v1/embeddings`. Uses `java.net.http` and Jackson. |
| `dukehoff-decide-spring-boot-starter` | Spring Boot 4 auto-configuration, `DecisionTask<T>` beans, health indicator, Micrometer metrics. |
| `dukehoff-decide-showcase` | End-to-end evaluation on public datasets: language models vs. trained classifiers (not published). |

## Spring Boot

```xml
<dependency>
    <groupId>com.dukehoff.decide</groupId>
    <artifactId>dukehoff-decide-spring-boot-starter</artifactId>
    <version>0.2.0</version>
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
default; for a trained classifier, declare
`@Bean ScoringBackend backend() { return LinearClassifier.load(...); }`. Tasks are created at startup, so an invalid output type or a missing required profile stops the
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
Options: `-Dshowcase.dataset=tickets|bitext|massive|cfpb` (massive needs `-Dshowcase.language=en` or `de`), `-Dshowcase.calibration=300`, `-Dshowcase.test=300`,
`-Dshowcase.language=all|en|de`, `-Dshowcase.seed=42`, `-Dshowcase.baseUrl=...`, `-Dshowcase.apiKey=...`.
`-Dshowcase.backend=linear` (with `-Dshowcase.train=3000`) needs no model: it trains a `LinearClassifier`
on examples outside both splits and writes the same report to `target/showcase/<dataset>-linear`;
`-Dshowcase.similarity=0.6` also drops training examples that resemble a calibration or test example.
`-Dshowcase.backend=embedding` trains an `EmbeddingClassifier` on embeddings from an OpenAI-compatible `/v1/embeddings`
endpoint (`-Dshowcase.embeddingUrl=http://localhost:8081/v1`, `-Dshowcase.embeddingModel=bge-m3`), for example
`llama-server -hf ggml-org/bge-m3-Q8_0-GGUF --embeddings --port 8081 -ub 8192 -b 8192`; vectors are cached in
`~/.cache/dukehoff-decide/embeddings`.
Datasets are downloaded on first run and keep their own licenses (tickets: CC BY-NC 4.0, Bitext: CDLA-Sharing-1.0,
MASSIVE: CC BY 4.0, CFPB: public domain).

## Build

```
mvn verify
```

Requires Java 21. Tests need no model; the live tests run only when `DUKEHOFF_DECIDE_OPENAI_BASE_URL` and
`DUKEHOFF_DECIDE_OPENAI_MODEL` are set. Every Java file needs the license header: `mvn license:format` adds it.

Releasing: see [RELEASING.md](RELEASING.md).

## License

[Apache License 2.0](LICENSE)
