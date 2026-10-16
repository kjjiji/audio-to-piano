# 钢琴改编助手

分析音频、分离声部、制作整首钢琴 MIDI 和试听，并根据原曲检查低音切分与重音。包含技能说明、可运行的 Python 脚本与校验流程，支持在具备命令执行能力的 AI 助手或 Linux 主机中使用。

公开源码：[kjjiji/111](https://github.com/kjjiji/111) · [下载安装包](https://github.com/kjjiji/111/releases/latest) · [运行说明](skills/audio-to-piano/references/runtime.md) · [测试记录](VALIDATION.md)

这是源码和技能插件；音频处理需要另行配置执行环境、模型与钢琴音色。GitHub 页面可以公开阅读和下载，当前没有在线上传歌曲并转换的网页服务。

## 获取与安装

下载 [Releases](https://github.com/kjjiji/111/releases/latest) 中的 `audio-to-piano-0.1.0.zip`，或克隆源码：

```bash
git clone https://github.com/kjjiji/111.git audio-to-piano
cd audio-to-piano
```

在支持本地技能的 Codex 中，将 `skills/audio-to-piano` 整个目录复制到用户技能目录，再按客户端提示重新加载技能：

```bash
mkdir -p ~/.codex/skills
cp -R skills/audio-to-piano ~/.codex/skills/
```

已有同名技能时先保留原目录，再手动更新。其他兼容客户端可使用完整插件包或技能目录，具体以客户端的导入能力为准。公开源码不会自动改变已有私人插件安装页的访问范围。

无需 AI 助手也可以直接使用下面的命令行入口。

## 使用示例

安装后可以直接说：

- “把这首歌做成纯钢琴，给我 MIDI 和试听。”
- “25 秒后左手切分不对，按原曲重新分析并修整。”
- “用这个 MIDI 做钢琴版，检查重复音和漏响。”
- “做一版去掉主旋律的钢琴伴奏。”

## 工作流程

原曲 → Demucs 六声部分离 → 基频/频谱/起音/拍点分析 → MuScriptor 或明确标注的 Basic Pitch 回退 → 钢琴声部整理 → MIDI 校验 → FluidSynth 合成 → 原曲同段对比 → 文件摘要与 ZIP。

输出保留原音频时间，避免固定根音重拍、鼓起音冒充和弦、低音泛音误判和最高音冒充旋律。支持任务缓存与指定阶段重做。纯器乐与有人声的音频走不同分支，已有 MIDI 可以直接生成钢琴试听。

## 运行要求

需要主机可执行 Python、FFmpeg/ffprobe 和 FluidSynth，并有可用模型、SF2 钢琴音色及足够内存/磁盘。插件不包含用户歌曲、模型权重、音色采样或凭证。手机网页可以发起任务和取得文件；计算由支持命令执行的主机完成。

已经验证 Linux/Python 3.11/CPU。先复用现有音频 Python 环境；新环境用隔离安装脚本：

```bash
python3.11 skills/audio-to-piano/scripts/bootstrap.py --venv /absolute/path/audio-piano-venv
/absolute/path/audio-piano-venv/bin/python skills/audio-to-piano/scripts/piano.py doctor \
  --model-path /absolute/path/model/model.safetensors \
  --soundfont /absolute/path/piano.sf2
```

模型旁保留匹配的 config.json；音色须覆盖输出实际使用的力度。系统依赖与模型获取见 [运行说明](skills/audio-to-piano/references/runtime.md)。不具备执行环境时，技能会说明缺失能力，不能自动在浏览器中运行本地脚本。

```bash
/absolute/path/audio-piano-venv/bin/python skills/audio-to-piano/scripts/piano.py run /absolute/path/song.mp3 \
  --work-dir /absolute/path/jobs/song --output-dir /absolute/path/output/song \
  --backend muscriptor --model-path /absolute/path/model/model.safetensors \
  --soundfont /absolute/path/piano.sf2
```

每首歌用独立任务目录。--force-from analyse 重做证据及后续，transcribe 重做转录，export 只重做编配与试听。--vocal-mode instrumental 可指定纯器乐；--accompaniment-only 省略主旋律层。完整参数用 piano.py run --help 查看。

已有 MIDI 时，可以保留音符时间和速度变化，使用钢琴音色合成：

```bash
/absolute/path/audio-piano-venv/bin/python skills/audio-to-piano/scripts/piano.py render-midi /absolute/path/song.mid \
  --work-dir /absolute/path/jobs/midi \
  --soundfont /absolute/path/piano.sf2
```

音频任务生成整首 MIDI、钢琴 MP3、原曲同段对比、分析报告与 ZIP。已有 MIDI 的合成任务没有原音频，不生成原曲对比。

## 校验与限制

检查 MIDI 音符读写一致性、时间、声道、重复琴键、SF2 力度覆盖、MP3 可解码、ZIP 与摘要。相邻同音、非 ASCII 标题、多速度 MIDI、低精度输入 MIDI 与音色空洞有回归测试。测试记录见 [VALIDATION.md](VALIDATION.md)。

自动转录仍可能漏音或误判。文件正确不等于音乐完全还原，默认版本也不保证双手可弹。遇到变速、复杂织体或用户指出的错拍，需要对照原曲再修整。

## 开发与打包

在具备运行依赖的隔离环境里执行：

```bash
python tests/test_regressions.py
python skills/audio-to-piano/scripts/piano.py --help
python tests/package_plugin.py . /tmp/audio-to-piano-0.1.0.zip
```

打包器会排除 Git 元数据和 Python 缓存，拒绝歌曲、MIDI、音色、模型和环境目录进入安装包。项目目录名称须为 `audio-to-piano`；下载 GitHub 自动生成的源码 ZIP 后，请将解压目录改成此名称再运行打包命令。GitHub Actions 只运行回归与打包检查，不下载模型或处理用户歌曲。

## 许可

本插件自己编写的代码与文档为 MIT。外部依赖、模型、音色与用户音乐遵循各自许可；见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。测试音乐不包含在插件包中。
