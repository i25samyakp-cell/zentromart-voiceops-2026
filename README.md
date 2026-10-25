# VoiceOps 2026 - ZentroMart Support Agent

**Participant:** Ishaan Agrawal  
**Public repository:** `REPLACE_WITH_PUBLIC_GITHUB_URL`

## What I built

I built a document-grounded customer-support agent for ZentroMart. It answers
policy questions, retrieves product facts, remembers a multi-turn conversation,
verifies a caller before exposing order information, uses the supplied order and
ticket tools, and connects to Gnani for speech-to-text and text-to-speech.

The implementation keeps the original section IDs with every retrieved chunk.
It prioritises current policies over guides, help articles, FAQs, and marketing
pages. It never uses the practice answers as runtime knowledge.

## How the system works

```text
Customer audio
  -> Gnani STT
  -> conversation and verification controller
  -> hybrid RAG over ZentroMart documents and catalog
  -> LLM grounded in retrieved evidence and verified order data
  -> Gnani TTS
  -> reply text, reply audio, sources, and tool trace
```

## Important setup choice

I recommend **Python 3.13**. Python 3.14 is newer than the versions explicitly
listed by several project dependencies and may cause installation problems.

## Baby-step setup on Windows

### 1. Open the project

Extract this ZIP, open the extracted folder in VS Code, and open a new terminal:

```text
Terminal -> New Terminal
```

### 2. Create a virtual environment

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
```

The terminal should now show `(.venv)` at the beginning of the line.

### 3. Install the packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the course dataset

Extract `Capstone_Project_Dataset_FTTT_2026.zip`. Either:

- rename the extracted folder to `dataset` and place it beside `run.py`; or
- keep it elsewhere and put its full path in `.env` as `DATASET_DIR`.

The dataset is intentionally not included in this submission package.

### 5. Create the private environment file

```powershell
Copy-Item .env.example .env
```

Open `.env` and add my own keys:

```text
OPENAI_API_KEY=my_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GNANI_API_KEY=my_gnani_key_here
DATASET_DIR=dataset
```

I never commit or submit `.env`.

### 6. Validate the project

```powershell
python validate_submission.py
```

This checks the supplied dataset, core retrieval logic, privacy boundary, and
required files. A missing API key is reported as an action item but is not
printed.

## Try the text agent

```powershell
python chat.py
```

Example questions:

```text
How fast is a UPI refund?
Can I return opened perfume?
I need help with order 8978 2837.
```

Type `quit` to finish.

## Required one-command voice run

Put my audio files in a folder named `input_audio`, then run:

```powershell
python run.py --input-audio input_audio --output outputs
```

For a multi-turn recording split into numbered clips, I use:

```powershell
python run.py --input-audio input_audio --output outputs --shared-session
```

For each input, the runner writes:

```text
outputs/001_example/
├── transcript.txt
├── response.txt
├── response.wav
├── retrieved_sources.json
├── tool_trace.json
└── result.json
```

## Evaluation

The practice answer fields are used only by this evaluation script, never by
the runtime agent.

Run all evaluations:

```powershell
python evaluate.py --suite all --output evaluation_outputs
```

This produces:

- a Step 1 baseline without documents;
- a Step 2 grounded-question evaluation;
- a retrieval evaluation;
- a Step 3 multi-turn conversation evaluation.

To test only local lexical retrieval without spending API credits:

```powershell
python evaluate.py --suite retrieval --lexical-only --output evaluation_outputs
```

After the API-backed evaluations complete, regenerate my report with the real
results:

```powershell
python make_report.py --evaluation-dir evaluation_outputs
```

## Design notes

### Retrieval

I chunk each Markdown file at its `[SECTION-ID]` headings and preserve its
document type, status, version date, and authority. Retrieval combines lexical
BM25-style scoring with optional OpenAI embeddings. Exact SKU and pincode rows
are added as structured chunks. The local lexical test retrieves at least one
gold source for every answerable practice question at top 8.

### Identity and tools

When an order ID first appears, I call `lookup_order(order_id)` without proof.
The response discloses no private order facts. I then ask for either the last
four registered phone digits or the delivery pincode, allow one retry, and only
store verified order data inside that conversation. Verification values are
redacted from tool traces and persisted transcripts.

I confirm a handover only after `create_ticket` returns `created=True`.

### Voice

I use Gnani only for STT and TTS. The default settings are `en-IN`, the Kaveri
voice, and `timbre-v2.5`. An empty transcript receives a short retry prompt.

## Repository structure

```text
.
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── knowledge.py
│   ├── llm.py
│   ├── tools_layer.py
│   ├── utils.py
│   └── voice.py
├── tests/
├── chat.py
├── evaluate.py
├── make_report.py
├── run.py
├── validate_submission.py
├── requirements.txt
├── .env.example
├── report.pdf
└── sample_outputs/
```

## Before I submit

I complete every item in `FINAL_SUBMISSION_CHECKLIST.md`, especially replacing
the repository URL, running the API-backed evaluation, regenerating the report,
recording the Gnani voice demo, and ensuring no keys are inside the ZIP.

