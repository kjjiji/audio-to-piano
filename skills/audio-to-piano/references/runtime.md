# 运行与安装

这是带脚本的技能插件，运行在可执行命令的主机里；手机浏览器本身不会运行 Python。主机需要本地文件读写、足够内存/磁盘，以及首次获取公开依赖和模型所需的网络。无执行环境时，保留插件使用说明并指出缺失能力，不能声称已处理音频。

## 已验证环境

原 0.1.0 完整流程：Linux x86_64、Python 3.11、PyTorch/Torchaudio 2.5.1 CPU、Demucs 4.1.0、MuScriptor 0.3.0、Beat This 1.1.0、Basic Pitch 0.4.0 ONNX、NumPy 1.26.4、FluidSynth 3、FFmpeg。新增 Python 3.12 安装与 CPU 运行检查见仓库 VALIDATION.md。GPU 安装方案与诊断入口已提供，GPU 实际推理和其他操作系统仍需验证。速度取决于时长与算力，不能许诺固定处理时长。

RTX 3090、Python 3.12 和已有 MuScriptor large 环境优先参考[GPU 环境复用与验证](gpu.md)。默认 CPU 安装仍保留，另有 `--profile cuda` 独立安装与 `--profile reuse` 补装模式；不要在现有 GPU 环境直接执行默认 CPU 安装。

先复用现有音频环境并运行 doctor；不修改系统 Python。新环境用：

```bash
python3.11 skills/audio-to-piano/scripts/bootstrap.py \
  --venv /absolute/path/audio-piano-venv
```

bootstrap 仅安装隔离环境里的 Python 依赖，不安装系统服务，也不下载歌曲、MuScriptor 权重或音色。--plan 可先打印执行计划。它保留主机已有代理和证书环境，不读取或打印凭证。Linux CPU 轮子使用 PyTorch 官方索引，Basic Pitch 使用 ONNX 路径，避免默认 TensorFlow 依赖。

FFmpeg/ffprobe 和 FluidSynth 共享库由主机包管理器提供。Debian/Ubuntu 常见包是 ffmpeg、libfluidsynth3。只在当前用户权限和主机规则允许时安装，不能提权绕过审批。在托管云执行环境中，按该环境的运行技能配置网络与证书。

## 模型与音色

- MuScriptor：--model-path 指向模型目录或 model.safetensors，旁边保留同一模型的 config.json。支持 small/medium/large，默认 medium。通过官方文档/模型仓库获取所需访问权限；--download-model 才允许加载接口获取尚未缓存的权重。插件不打包或重新授权权重。
- Demucs htdemucs_6s：首次运行使用上游的公开下载机制，之后复用本机缓存。
- Beat This：--beat-checkpoint 可显式指定 final0 checkpoint，未指定则由上游公开机制加载。变更 checkpoint 后从 analyse 重做。
- 钢琴 SF2：--soundfont 或环境变量 PIANO_SOUNDFONT 指定音色。必须存在 bank 0/program 0 钢琴，并覆盖输出实际使用的琴键和力度。不能仅按文件名推断所有权或许可；提供许可和 --soundfont-credit。

已验证的 Salamander Grand Piano 是 Alexander Holm 的 Yamaha C5 采样，CC BY 3.0：https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html#SalamanderGrandPiano 。本会话使用的连续力度改版调整了 zone 映射，采样未改变，交付时注明修改。原某个 6 层映射版本有力度空洞，不可为了消除漏响把所有音符强制设成同一力度。也可用其他经校验且许可适合的 SF2；不要对所有 SF2 套用 Salamander 的署名。

自动查找仅检查常见用户缓存、当前目录的 scratch 缓存、FUFUMIDI_MODELS_DIR 和系统音色目录；插件没有固定歌曲路径。迁移到新环境建议显式指定模型和音色。

```bash
/absolute/path/audio-piano-venv/bin/python skills/audio-to-piano/scripts/piano.py doctor \
  --model-path /absolute/path/model/model.safetensors \
  --soundfont /absolute/path/piano.sf2
```

doctor 现在实际导入依赖、检查配套版本、执行所选设备的小型算子、创建 CPU ONNX session，并检查模型配置、音色覆盖与实际合成。`ready` 综合所选后端、依赖、设备与音色；`--strict` 不通过时退出 1。默认仍输出诊断并退出 0。MuScriptor 模型默认不加载，`--smoke-test` 才进行本地模型推理；只有 GPU 推理通过才将 `gpu_inference_verified` 设为 true。`ready` 本身不证明已完成模型推理或音乐正确。报告中 Basic Pitch ONNX (fallback) 是回退引擎，必须如实说明。

## 缓存与安全边界

每个 work-dir 保存 config.json、源文件摘要、分离 WAV、独立证据 NPZ、转录 MIDI、阶段标记和日志；output-dir 保存交付文件。音频缓存可能很大。只有用户要求清理时才删除其任务数据；不要打包到插件里。

所有外部命令使用参数列表，文件名不作为 shell 代码。调用者也要正确引用文件路径，不能把用户文件名插入未转义的 shell 字符串。与输入相关的本地路径仅用于任务内部；发布报告尽量只保留文件名、时长和摘要。
