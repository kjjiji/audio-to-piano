# RTX 3090：复用环境、安装与验证

本页适用于 Linux x86_64、Python 3.11/3.12。优先使用服务器上已经能运行 MuScriptor large 的解释器和本地权重。当前改动已执行两种 Python 的 CPU 验证；RTX 3090、服务器驱动和 large 权重仍需要按下面步骤实测。`dist/audio-to-piano-0.1.0.zip` 保留原发布内容，这些新增功能需从包含本页的源码运行。

## 兼容性判断

| 项目 | 处理方式 |
|---|---|
| Python 3.12 | 支持安装；原 Essentia `2.1b6.dev1110` 没有 cp312 wheel，改用有 Linux cp312 wheel 的 `2.1b6.dev1389`。Python 3.11 保留原版本 |
| 已有 PyTorch/MuScriptor | 先诊断，能用就直接运行。补装模式固定所有已有 distribution 的版本，仅安装缺失项目，冲突时停止解析 |
| 独立 GPU 环境 | 固定 `torch==2.5.1+cu124` 和 `torchaudio==2.5.1+cu124`，两者从同一官方 CUDA 12.4 索引安装 |
| 服务器 CUDA 13.2 | `nvidia-smi` 显示的是驱动支持的 CUDA 上限，`nvcc --version` 才是本机 toolkit。wheel 自带用户态 CUDA 运行库，不必安装同版本 toolkit；较新驱动可运行较旧 CUDA 构建，但最终以实际算子和模型测试为准 |
| RTX 3090 精度 | `auto` 在所选设备原生支持 BF16 时用 BF16，否则 FP16；CPU 保留 BF16。可显式选 FP32 做数值对照 |
| Basic Pitch | 继续使用 CPU ONNX，避免 Python 3.12 下上游默认 TensorFlow `<2.15.1` 依赖无法安装的问题。已有 `onnxruntime-gpu` 时复用其 CPU provider |
| 其他阶段 | Demucs 与 MuScriptor 使用 `--device` 指定的设备；Beat This、独立音高证据、Basic Pitch ONNX、编配和 FluidSynth 仍使用 CPU |

依据：[PyTorch 2.5.1 官方安装命令](https://pytorch.org/get-started/previous-versions/#v251)、[TorchAudio 版本配套与 Python 支持](https://docs.pytorch.org/audio/2.5.0/installation.html)、[NVIDIA 驱动兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[Essentia 3.12 wheel](https://pypi.org/project/essentia/2.1b6.dev1389/#files)。这些发行资料说明存在可选组合，不代表该服务器已实测。

## 1. 先检查已有环境

在仓库根目录运行；只需替换下面四个路径。`PY` 必须是现有 MuScriptor 环境的 Python，不是系统 Python。`MODEL` 可以是目录或权重文件，旁边需有匹配的 `config.json`。

```bash
PY=/absolute/path/muscriptor-env/bin/python
MODEL=/absolute/path/muscriptor-large/model.safetensors
SF2=/absolute/path/piano.sf2
SONG=/absolute/path/song.mp3
CLI=skills/audio-to-piano/scripts/piano.py

"$PY" "$CLI" doctor --device cuda:0 --backend muscriptor --model-size large \
  --model-path "$MODEL" --soundfont "$SF2" --strict > doctor-gpu.json
```

输出包括 Python 和依赖版本、真实导入错误、`pip check` 冲突、PyTorch CUDA runtime、TorchAudio 配套、GPU 名称/架构/显存、驱动和 toolkit 信息、所选设备/精度的张量分配、矩阵乘法与注意力算子检查、模型配置、ONNX session、音色 bank 0/program 0、88 键全力度覆盖及实际合成。

`runtime.ready` 只说明小型算子通过；`model.ready` 只说明本地权重存在且配置字段有效，未证明权重与配置匹配。只有实际模型推理通过后，`gpu_inference_verified` 才为 `true`。`ready` 综合所选后端、依赖、设备和音色；`--strict` 不通过则退出码为 1。默认 doctor 仍输出 JSON、退出 0，便于在尚未配齐资源的环境收集诊断。拍点 checkpoint 未缓存时单独提示，不把它描述为已加载；运行流程可能下载 final0。

本项目有意省略 Basic Pitch 的 TensorFlow 默认依赖；doctor 仅豁免该条“未安装 TensorFlow”的 `pip check` 提示，并另建 ONNX session 检查可用性。其他冲突仍会报错。doctor 不下载 MuScriptor 权重或修改环境。

## 2. 只补装缺失依赖

已有环境只有在 doctor 提示依赖缺失时才需要此步骤。先查看安装计划：

```bash
ENV_DIR=$("$PY" -c 'import sys; print(sys.prefix)')
"$PY" skills/audio-to-piano/scripts/bootstrap.py \
  --profile reuse --venv "$ENV_DIR" --plan > reuse-plan.json

"$PY" skills/audio-to-piano/scripts/bootstrap.py --profile reuse --venv "$ENV_DIR"
```

`--plan` 读取目标解释器的版本和已安装包，不写文件、不安装或下载。执行时生成约束文件，先做 pip dry-run，再安装缺失依赖；不升级 pip，不替换已有 torch、torchaudio、MuScriptor、NumPy 或其他已安装包，结束时再次核对原版本。这个模式不会自动修复已存在的损坏安装或版本冲突；遇到无法兼容的组合，请使用新环境。

## 3. 必要时建立独立 GPU 环境

目标目录必须尚未创建，避免覆盖原 MuScriptor 环境。此方案不更换系统驱动、不编译 CUDA 扩展，也不下载 MuScriptor 权重或音色。

```bash
GPU_ENV="$HOME/.venvs/audio-piano-cu124"
python3.12 skills/audio-to-piano/scripts/bootstrap.py \
  --profile cuda --python python3.12 --venv "$GPU_ENV" --plan
python3.12 skills/audio-to-piano/scripts/bootstrap.py \
  --profile cuda --python python3.12 --venv "$GPU_ENV"
PY="$GPU_ENV/bin/python"
```

FFmpeg/ffprobe 和 FluidSynth 共享库仍由主机提供。安装完成时的 doctor 是诊断输出，不能将安装命令退出 0 当成 GPU 验证成功，必须继续执行下面带 `--strict` 的测试。

## 4. 验证 large 真正进行 GPU 推理

```bash
"$PY" "$CLI" doctor --device cuda:0 --dtype bfloat16 \
  --backend muscriptor --model-size large --model-path "$MODEL" --soundfont "$SF2" \
  --smoke-audio "$SONG" --strict > doctor-gpu-inference.json

"$PY" - <<'PY'
import json
r=json.load(open('doctor-gpu-inference.json'))
assert r['ready'] and r['gpu_inference_verified'], r
assert r['model_inference']['completed_chunks'] >= 1, r
print(r['runtime']['devices'])
print(r['model_inference'])
PY
```

该命令只读取歌曲前 5 秒，执行包含钢琴与低音的多乐器转录并解析 MIDI，不保存歌曲副本或模型。省略 `--smoke-audio` 并指定 `--smoke-test` 时用临时合成信号。两种方式都验证推理能否执行，不评估错音率，也不代替整条流水线测试。

MuScriptor 0.3.0 内部会固定启用 CUDA FP16 autocast。本适配层在选择 BF16/FP32 时将该上下文与请求精度对齐，条件编码仍按上游保持 FP32；FP32 同时关闭 TF32。此处兼容层使用 `_model.autocast`，已有 MuScriptor 版本若改变该内部接口，会明确报错，不会悄悄用其他精度运行；可改用 `--dtype float16` 或独立的 0.3.0 环境验证。

## 5. 验证完整 GPU 流水线

先用同一首歌的短片段检验 Demucs、转录、编配、MIDI 导出和音频试听，再运行整首。首次运行可能通过各上游机制下载 Demucs 和 Beat This 公共权重。片段必须有清晰节拍；12 秒只是启动测试，不是整曲交付。

```bash
GPU_CHECK_DIR=$(mktemp -d -t audio-piano-gpu-XXXXXXXX)
ffmpeg -v error -y -i "$SONG" -t 12 -ar 44100 -ac 2 "$GPU_CHECK_DIR/excerpt.wav"
"$PY" "$CLI" run "$GPU_CHECK_DIR/excerpt.wav" \
  --work-dir "$GPU_CHECK_DIR/job" --device cuda:0 --dtype bfloat16 \
  --backend muscriptor --model-size large --model-path "$MODEL" --soundfont "$SF2" \
  > "$GPU_CHECK_DIR/run.jsonl"
printf '%s\n' "$GPU_CHECK_DIR"
```

检查任务目录的 `separate.log`、`transcribe-*.log`、`config.json` 以及 `output/analysis_report.json`、`output/file_manifest.json`、MIDI、MP3、同段原曲对比与 ZIP。GPU 上各转录角色仍串行执行，避免同时载入多个 large 模型。large 首次加载还会暂时使用 FP32 权重，不能只根据最终 BF16 参数量估算峰值显存。CUDA 或显存错误会保留日志并失败；不会静默改用 CPU、较小模型或 Basic Pitch。

短片段通过后用新任务目录处理整首：

```bash
"$PY" "$CLI" run "$SONG" --work-dir "$HOME/piano-jobs/song-large-gpu" \
  --device cuda:0 --dtype bfloat16 --backend muscriptor --model-size large \
  --model-path "$MODEL" --soundfont "$SF2"
```

## 音符质量与数值对照

GPU 支持本身不保证减少错音。显式 `--backend muscriptor` 确保使用 large 而不会自动回退；转录仍分低音、器乐伴奏和必要的人声，保留 `prelude_forcing=True`、跨窗口衔接、独立基频/起音证据与现有异常音符筛选。推理显式关闭采样，避免把随机变动当成修复。

如果低精度输出出现异常，可在同一短片段、同一权重与参数下用 `--dtype float32` 对照，先确认显存足够。设备、精度、MuScriptor 版本或模型配置改变会使转录缓存失效；可用 `--force-from transcribe` 明确重做。对照 MIDI 的音域、重复音、音长、起音和原曲试听，再决定精度；FP32 也不等于模型已正确识谱。细节见[修整流程](refinement.md)。

## 不依赖 GPU 的开发检查

```bash
"$PY" tests/test_regressions.py
"$PY" tests/test_runtime.py
"$PY" tests/test_runtime_integration.py
```

第三条需要完整依赖、FFmpeg 和可用 SF2；可用 `TEST_PIANO_SOUNDFONT` 指定。它使用临时生成的微型随机 MuScriptor 模型测试 doctor 和转录 worker 的真实 CPU 执行，另测 Basic Pitch ONNX、Essentia、MIDI 合成、清单和摘要，不下载正式模型。GPU 分支的单元模拟不能替代前面的 RTX 3090 实机测试。
