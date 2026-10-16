# Validation record — 0.1.0

## Public-source packaging check — 2026-10-16

The public-source copy retains the 0.1.0 transcription pipeline. Its README now includes source download, local-skill installation and standalone CLI usage; the packager also excludes Git metadata so it works from a normal clone. The public `dist/` directory holds only the downloadable plugin ZIP and its checksum, and is excluded from the package itself.

All five regression tests passed again on Python 3.11.17. The CLI help and runtime doctor passed using the existing configured environment and external model/font assets. The package passed the official plugin JSON schema, Python syntax, contained-path and archive-byte checks with a local `.git` directory present. No user media, task caches, model weights, soundfonts or credentials are included. These publication checks did not rerun full-song transcription or measure musical accuracy. The integration results below are the earlier 2026-10-13 runs.

## Original pipeline validation — 2026-10-13

Validated on 2026-10-13 in Linux x86_64, Python 3.11.17, PyTorch/Torchaudio 2.5.1 CPU, four CPU threads, FFmpeg and libFluidSynth. Validation media, cached model weights, fonts, task logs and generated audio are not included in the plugin archive.

| Case | Result |
|---|---|
| Complete 287.191-second source, import of existing source-matched MuScriptor cache | 4,426 notes / 4 piano tracks; fresh arrangement, MIDI, full audio, A/B comparison and ZIP export passed |
| New 12-second vocal excerpt, full CLI pipeline including separation and direct MuScriptor calls | 223 notes / 4 tracks; all output checks passed |
| New 12-second instrumental excerpt, full CLI pipeline | 282 notes / 4 tracks including instrumental primary melody; all output checks passed |
| Basic Pitch ONNX fallback on the instrumental excerpt, reuse of source-matched separation only | 267 notes / 4 tracks; independent analysis, fallback transcription, export and engine label passed |
| Supplied MIDI with two tempo regions, UTF-8 names, shell metacharacters in the filename and drum notes | 4 pitched notes preserved, 2 drum notes removed, increased PPQ with no timing change; piano audio and ZIP passed |
| Same input/configuration run again | Cache hit; output SHA-256 values unchanged |
| Different input pointed at an existing job | Rejected before changing that job's configuration |
| Focused comparison at source time 25 seconds, length 12 seconds | Correct source cues; separate output; complete A/B file unchanged |
| Real Salamander SF2 coverage | Continuous mapping covered all 88 keys at velocities 1–127; comparison mapping with gaps had 2,640 uncovered key/velocity pairs |

The maximum MIDI serialization timing error in the audio cases was 0.000224 seconds. Actual output manifest byte counts and SHA-256 values matched files. Exporters decoded complete MP3 files, tested ZIP CRCs and verified embedded file bytes.

Five regression tests pass in tests/test_regressions.py: touching repeated keys retain the second duration; UTF-8 metadata and late-song timing survive; variable tempo survives increased PPQ; cross-track identical-key collisions are rejected; a synthetic SoundFont velocity gap cannot silently remove required notes.

```bash
python tests/test_regressions.py
python tests/package_plugin.py /absolute/path/audio-to-piano /absolute/path/audio-to-piano-0.1.0.zip \
  --schema /absolute/path/official-plugin.schema.json
```

The installer plan and pinned requirement resolution were checked against the existing compatible runtime. A complete clean environment installation, GPU execution, macOS and Windows were not exercised. The integration runs used the existing configured model and font assets.

These are execution and data-integrity checks. They do not establish note-by-note musical fidelity, listening approval or two-hand playability. The long-song case reused previous neural transcription; it did not rerun long-song inference. The new excerpt cases exercised the new entry point and inference wrappers.
