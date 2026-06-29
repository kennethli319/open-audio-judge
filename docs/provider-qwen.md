# Local Qwen/Qwen3-Omni Provider

Open Audio Judge's first real provider targets Qwen/Qwen3-Omni served behind an OpenAI-compatible chat-completions endpoint.

## Environment

```bash
export OAJ_PROVIDER=qwen
export OAJ_BASE_URL=http://localhost:8091/v1
export OAJ_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
export OAJ_API_KEY=EMPTY
```

## Request Shape

The provider sends:

```json
{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [
      {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
      {"type": "text", "text": "..."}
    ]}
  ],
  "modalities": ["text"],
  "temperature": 0.0
}
```

If the case has `audio_url`, the URL is passed through. If it has `audio_path`, the file is encoded as a data URL. Some local servers may require remote HTTP URLs instead of data URLs; in that case, host your audio files locally and put those URLs in the case file.

## Why `modalities: ["text"]`

For judging, we only need structured text output. vLLM-Omni documents modality control for Qwen3-Omni and supports text-only output through `modalities: ["text"]`.

## Current Limits

- Long audio clips should be segmented before judging.
- The MVP does not resample audio; prepare files in a format your model server accepts.
- JSON-mode support varies across OpenAI-compatible servers, so Open Audio Judge parses and validates the model text instead of relying only on server-side `response_format`.
