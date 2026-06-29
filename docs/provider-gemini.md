# Gemini Provider

Open Audio Judge supports a `gemini` provider for hosted audio-capable judging through the Gemini Interactions API.

The provider follows the current Gemini audio understanding pattern:

- `POST /v1beta/interactions`
- `model: gemini-3.5-flash`
- `input` parts containing text rubric, required audio, and case prompt
- inline audio data for local files, or `uri` for URL audio
- `response_format` populated from the prompt's JSON schema when present

Source: https://ai.google.dev/gemini-api/docs/audio#python

## Usage

Set the API key at runtime. Do not put API keys in Git, prompt YAML files, reports, or case files.

```bash
export OAJ_PROVIDER=gemini
export GEMINI_API_KEY=...

oaj eval \
  --provider gemini \
  --judge asr_error \
  --cases examples/asr_cases.jsonl \
  --out runs/gemini-asr
```

Optional overrides:

```bash
export OAJ_MODEL=gemini-3.5-flash
export OAJ_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

For local audio paths, the provider sends inline base64 audio. For remote examples, use `audio_url` so Gemini receives a URI.

Gemini cases must include both audio (`audio_path` or `audio_url`) and textual context
(`reference_text`, `candidate_text`, or `turns`). This keeps every hosted judge call grounded in
the sample audio and in the task text the researcher wants evaluated.
