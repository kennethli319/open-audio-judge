# Development Audio Samples

These are small, open-source/free-license audio samples for local development and hosted judge smoke tests. They are linked, not vendored, so the repository stays lightweight.

Use these only as development fixtures. Real benchmark releases should pin exact files, licenses, checksums, and transcripts in a dataset card.

## ASR Error Samples

- `asr-open-armstrong-small-step`: Neil Armstrong/NASA Apollo 11 quote. Source page: https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/Armstrong%20Small%20Step.ogg. License note: public domain, NASA.
- `asr-open-en-us-hello`: US English pronunciation of "hello". Source page: https://commons.wikimedia.org/wiki/File:En-us-hello.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-hello.ogg. License note: Wikimedia Commons free media page; verify exact license before redistributing.
- `asr-open-en-uk-hello-1`: UK English pronunciation of "hello". Source page: https://commons.wikimedia.org/wiki/File:En-uk-hello-1.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/En-uk-hello-1.ogg. License note: public domain dedication on source page.

Manifest: `examples/asr_open_samples.jsonl`.

## TTS Naturalness Samples

- `tts-open-armstrong-small-step`: human historical speech baseline for naturalness and artifact handling. Source page: https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/Armstrong%20Small%20Step.ogg.
- `tts-open-en-us-hello`: short US English single-word naturalness baseline. Source page: https://commons.wikimedia.org/wiki/File:En-us-hello.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-hello.ogg.
- `tts-open-en-uk-hello-1`: short UK English single-word naturalness baseline. Source page: https://commons.wikimedia.org/wiki/File:En-uk-hello-1.ogg. Audio URL: https://commons.wikimedia.org/wiki/Special:Redirect/file/En-uk-hello-1.ogg.

Manifest: `examples/tts_open_samples.jsonl`.

## Local WAV Fixtures

Generate local 16 kHz mono WAV fixtures under ignored `runs/`:

```bash
python scripts/materialize_open_samples.py
```

This writes:

- `runs/open-audio-samples/wav/*.wav`
- `runs/open-audio-samples/asr_wav_cases.jsonl`
- `runs/open-audio-samples/tts_wav_cases.jsonl`

Use those derived manifests when a provider works more reliably with inline local audio than remote URLs.

## Gemini Sample Records

Gemini smoke results for these development samples are committed in `examples/gemini_sample_records.jsonl`.
Before rerunning the same samples, check whether the recorded result is still current:

```bash
python scripts/gemini_sample_records.py check
```

This check fingerprints the sample manifest data, judge id/version, provider, and model. Rerun Gemini only when the check reports a missing/changed sample, such as after editing an open sample manifest or judge prompt.

## Running A Smoke

```bash
oaj eval --provider mock --judge asr_error --cases examples/asr_open_samples.jsonl --out runs/asr-open-samples
oaj eval --provider gemini --judge tts_naturalness --cases examples/tts_open_samples.jsonl --out runs/tts-open-samples
oaj eval --provider gemini --judge asr_error --cases runs/open-audio-samples/asr_wav_cases.jsonl --out runs/gemini-asr-open-wav-samples
oaj eval --provider gemini --judge tts_naturalness --cases runs/open-audio-samples/tts_wav_cases.jsonl --out runs/gemini-tts-open-wav-samples
python scripts/gemini_sample_records.py update --results runs/gemini-asr-open-wav-samples/results.jsonl --results runs/gemini-tts-open-wav-samples/results.jsonl
```

Gemini runs require `GEMINI_API_KEY` to be supplied at runtime.
