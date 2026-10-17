# Validation record

## GPU runtime adaptation (source, not a new 0.1.0 ZIP) — 2026-10-17

Baseline: commit `3718aa7`, with the successful [original Actions run](https://github.com/kjjiji/audio-to-piano/actions/runs/38848342131). That run exercised regression and packaging checks, not GPU inference. The original integration results below remain historical CPU results.

The current validation host is Linux x86_64 with **no NVIDIA device, driver tooling, CUDA toolkit, RTX 3090 connection or deployed large weights**. All tests below were actually executed on this host. Model/font/media fixtures stayed outside the repository and package.

| Check | Actual result |
|---|---|
| Fresh default CPU bootstrap, Python 3.11.17 | Completed in a new venv; torch/torchaudio `2.5.1+cpu`, original Essentia `2.1b6.dev1110`, all required imports passed |
| Python 3.12.14 CPU dependency installation | Completed in a new venv; torch/torchaudio `2.5.1+cpu`, Essentia `2.1b6.dev1389`, all required imports passed |
| Fresh `--profile cuda` bootstrap, Python 3.12.14 | Completed in a separate new venv; torch/torchaudio `2.5.1+cu124` imported, `torch.version.cuda == "12.4"`; this verifies installation, **not GPU execution** |
| Existing CUDA-wheel environment, `--profile reuse` | Completed and checked that every previously installed distribution retained its version; CUDA wheels were not replaced by CPU wheels |
| Existing Python 3.12 CPU environment with pyfluidsynth deliberately removed | Reuse plan and pip dry-run selected the missing dependency, installed pyfluidsynth, and verified all original distribution versions were preserved; doctor returned ready |
| Original MIDI/SF2 regressions | All 5 passed under both Python 3.11 and 3.12 |
| New runtime tests | All 15 passed under both Python versions: explicit CUDA rejection, GPU index/native BF16 selection, kernel failure, TorchAudio mismatch, autocast alignment, model config checks, HF symlinks, cache invalidation, installer preservation, strict exit codes and dependency exclusions |
| Offline CPU integration | All 3 passed under both Python versions: actual doctor plus worker inference, Basic Pitch ONNX plus Essentia execution, and real MIDI → FluidSynth → MP3/ZIP export with manifest SHA-256 checks |
| Local MuScriptor inference fixture | A temporary **random, untrained** 32-dimension / 1-layer model with 1,395-token vocabulary completed a 5-second multi-instrument CPU inference chunk through the real public API; parsed Type 1 MIDI was emitted. The model is an execution fixture, **not small/medium/large or an accuracy benchmark** |
| Actual SF2 and evidence checks | System TimGM6mb bank 0/program 0 covered all 88 keys at velocities 1–127 and synthesized non-silent audio; Basic Pitch CPU ONNX session and prediction ran; Essentia Melodia produced finite arrays |
| CUDA requested on CPU-only torch | `doctor --strict --device cuda` exited 1 with a no-CUDA-build error and retained model/font diagnostics |
| CUDA requested on installed CUDA torch without hardware | `doctor --strict --device cuda:0` exited 1 with CUDA unavailable, no devices, kernel test not run and `gpu_inference_verified: false`; no CPU fallback was reported as GPU success |
| Plugin packaging | Official JSON schema, Python syntax, contained paths, ZIP CRC and archived bytes passed; no neural weights, user media, fonts or venvs included |

`pip check` has one expected exclusion: Basic Pitch 0.4.0 declares a TensorFlow dependency which is intentionally omitted for the ONNX workflow. Doctor lists that exact missing-TensorFlow message separately, checks an actual CPU ONNX session, and still rejects other dependency conflicts.

The unit CUDA cases use mocks; all actual model and audio execution above used CPU. **Not verified:** NVIDIA driver/CUDA 13.2 compatibility on the user's server; real RTX 3090 GEMM/attention/BF16/FP32; large-model loading, inference, peak VRAM or speed; GPU Demucs and the complete GPU song pipeline; production-weight musical accuracy or reductions in wrong notes. No fresh full-song transcription with published MuScriptor weights was performed in this session. BF16/FP32 autocast handling uses the guarded MuScriptor 0.3.0 `_model.autocast` interface and must also be checked against the reused server version.

Reproduction with an installed runtime (run from repository root):

```bash
python tests/test_regressions.py
python tests/test_runtime.py
TEST_PIANO_SOUNDFONT=/absolute/path/piano.sf2 python tests/test_runtime_integration.py
python tests/package_plugin.py . /tmp/audio-to-piano-gpu-support.zip \
  --schema /absolute/path/official-plugin.schema.json
```

The Actions workflow now has Python 3.11/3.12 regression/package jobs and separate fresh CPU-bootstrap/offline-integration jobs. These jobs do not have GPU hardware or published MuScriptor weights. For copyable **server-side** reuse, GPU installation, strict CUDA/large inference and full pipeline commands, see [RTX 3090 validation](skills/audio-to-piano/references/gpu.md).

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
