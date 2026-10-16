# 交付规则

音频输入交付整首 Piano.mid、Piano.mp3、Piano-Source-Comparison.mp3、analysis_report.json 和 ZIP。ZIP 内有 README 与音色署名；file_manifest.json 记录当前版本文件的大小和 SHA-256。WAV、分离声部和模型日志作为任务缓存，不默认塞进插件或下载包。

同段对比按“钢琴 → 一秒间隔 → 原曲”排列，报告给出源时间和对比文件中的起始点；用统一响度便于比较节奏与音符。它不是无间隔同步混音，也不替代听验。compare 生成指定时间段的独立文件，不覆盖全曲对比。

完成后检查：

1. MIDI 能重新读取，音高/力度/数量一致，最大起止误差小于 1 ms。
2. 每个实际输出音符在 SF2 bank 0/program 0 都有覆盖，没有静音力度区间。
3. 成品音频非空且不静音，FFmpeg 全文件解码成功；检查峰值与时长。
4. ZIP testzip 通过，内嵌文件字节与独立文件一致，文件清单摘要正确。
5. 链接由真实主机上传回执生成，指向本次输入的当前版本。

使用当前主机原生文件交付能力。若采用 ChatGPT Pages，创建或使用任务授权的私有 Page，先读页面指导，再 write_page_reference 上传实际本地路径，核对 byte_size、sha256 和 file_access_confirmed。上传本身不插入页面；用 read_page 的新鲜块、哈希、stream_kind 和存在时的 content.through_sequence，再 edit_page 插入回执提供的 Markdown，检查每个操作状态并读回。未知写入状态先核对现状，不盲目重试上传或插入。

Native Files 单文件限制可能为 10 MiB。默认码率会根据曲长调整；仍超限时使用平台支持的文件方式，或制作合理分包并明确各包内容。不得编造执行环境公网地址、sandbox:/mnt/data 地址或沉积文件 ID 来当下载入口。不得未经授权把用户音乐上传到公开 file.kiwi、GitHub release 或其他公开网站。

交付说明简洁列出所用转录引擎、整体产物和未完成的音乐判断。不要宣称已经人工听验、完全保真或必定双手可弹。用户反馈目标段时继续迭代，保留版本对应关系。
