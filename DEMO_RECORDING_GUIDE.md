# Two-minute demo recording guide

I will record this only after both API keys work and `run.py` has produced real
Gnani audio. I will not submit a simulated or manually edited agent answer.

## Before recording

1. I choose one supplied practice order whose verification value I am allowed to
   use because all records are synthetic.
2. I start `python chat.py` for an easy on-screen demonstration, or my voice UI
   if I have built one.
3. I confirm that Gnani handles both my microphone input and spoken output.
4. I hide `.env`, API keys, phone data, and any verification value from the
   screen recording.

## Suggested sequence

### 0:00-0:15 - Introduce the project

“I built a voice-enabled support agent for ZentroMart. It uses Gnani for speech,
retrieves evidence from company documents, remembers the conversation, and can
use verified order tools.”

### 0:15-0:55 - Grounded policy answer

Ask a policy question, such as:

```text
How quickly does a UPI refund arrive after it is initiated?
```

Show the spoken reply and briefly reveal `retrieved_sources.json`, including
`REF-2` rather than the weaker generic FAQ.

### 0:55-1:30 - Follow-up with context

Ask a follow-up:

```text
What if I paid by card instead?
```

Show that the agent understands “instead” without needing the entire question
again.

### 1:30-1:55 - Safe handover or unknown answer

Use one of these:

```text
Is there a student discount?
```

The correct behavior is to say the documents do not contain that information
and offer a human colleague.

Alternatively, demonstrate a required escalation and show that a ticket is
confirmed only after `create_ticket` succeeds.

### 1:55-2:00 - Close

“The same agent handles text and voice, and every response saves its evidence and
tool trace for reproducible testing.”

## Final checks

- The video is no longer than two minutes.
- My own voice is used with consent.
- Gnani is the only STT and TTS provider.
- No answer was typed or edited by hand.
- No secret or verification value is visible.

