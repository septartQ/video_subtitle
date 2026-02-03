# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在操作此代码仓库时的指导。

## 项目概述

这是一个基于 Python 的视频自动字幕生成与翻译工具。它从视频中提取音频，使用 Whisper 语音识别生成字幕，通过阿里云百炼 API 翻译成中文，并将字幕嵌入到视频中。

**语言要求**：使用中文回复用户，但编写代码时使用英文。

## 开发命令

### 环境设置

```bash
# 创建虚拟环境（需要 Python 3.12）
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
uv pip install -e .
# 或
uv pip install -r requirements.txt
```

### 运行应用

**使用 `uv run`（推荐 - 无需手动激活虚拟环境）：**
```bash
# 完整流程（语音识别 + 翻译 + 嵌入）
uv run python video_subtitle.py input.mp4

# 测试 API 连通性
uv run python video_subtitle.py --test-api

# 常用选项
uv run python video_subtitle.py input.mp4 -o output.mp4                    # 指定输出路径
uv run python video_subtitle.py input.mp4 --model base                     # 使用更小/更快的模型
uv run python video_subtitle.py input.mp4 --language en                    # 指定源语言
uv run python video_subtitle.py input.mp4 --audio-segment 15               # 减小分段长度（显存不足时）
uv run python video_subtitle.py input.mp4 --batch-size 20                  # 减小批量大小（token 限制时）
uv run python video_subtitle.py input.mp4 --skip-translation               # 仅生成原文字幕
uv run python video_subtitle.py input.mp4 --skip-embedding                 # 不嵌入字幕到视频
uv run python video_subtitle.py input.mp4 --skip-transcription             # 使用已有字幕
uv run python video_subtitle.py input.mp4 --no-cache                       # 禁用翻译缓存
```

**使用已激活的虚拟环境：**
```bash
python video_subtitle.py input.mp4
```

### 可用的 Whisper 模型

- `tiny`、`base`、`small`、`medium`、`large-v1/v2/v3`（默认：large-v3）

## 架构

### 单文件架构

所有核心功能都包含在 `video_subtitle.py` 中（约 1150 行）。代码按以下类组织：

```
Config（数据类）
├── 百炼 API 配置（从 .env 读取 DASHSCOPE_API_KEY）
├── Whisper 配置（模型、设备、计算类型）
├── 音频分段设置（AUDIO_SEGMENT_MINUTES）
├── 翻译缓存设置
└── 字幕/视频编码设置

TranslationCache
├── 基于 SQLite 的缓存，使用 MD5 hash 作为键
└── 线程安全的连接管理

AudioExtractor
├── extract_segment() - 提取指定时间段音频
└── extract() - 自动分段长视频

SubtitleGenerator
├── load_model() - 延迟加载 Whisper 模型
├── transcribe_audio() - 使用 VAD 识别语音
└── generate() - 合并多段结果并调整时间戳

SubtitleTranslator
├── translate_batch() - 批量翻译并检查缓存
├── _translate_one_by_one() - 行数不匹配时的备用方案
└── translate_srt() - 切片处理整个 SRT 文件

VideoEmbedder
└── embed() - 使用 FFmpeg 硬编码字幕并显示进度

VideoSubtitlePipeline
└── process() - 编排完整工作流程
```

### 关键设计模式

1. **延迟加载**：faster_whisper 和 requests 在方法内部导入，而非模块级别
2. **自动降级**：CUDA 不可用时自动切换到 CPU
3. **分段处理**：长视频被切分成块（默认 30 分钟）以避免内存不足
4. **批量翻译**：每次翻译 30 行以优化 API 使用
5. **备用机制**：批量翻译返回的行数不匹配时，回退到逐条翻译

### 配置

从 `.env` 文件加载环境变量：
- `DASHSCOPE_API_KEY` - 翻译 API 必需

关键配置默认值：
```python
WHISPER_MODEL = "large-v3"
DEVICE = "cuda"  # 不可用时自动切换到 CPU
COMPUTE_TYPE = "float16"  # CPU 模式下切换为 int8
AUDIO_SEGMENT_MINUTES = 30
TRANSLATION_BATCH_SIZE = 30
VIDEO_CODEC = "h264_nvenc"  # CPU 模式下切换为 libx264
```

### 输出文件

```
temp/
├── translation_cache.db              # SQLite 翻译缓存
├── {video_name}_original.srt         # 源语言字幕
└── {video_name}_translated.srt       # 中文字幕

{video_dir}/
└── {video_name}_subtitled.mp4        # 最终输出
```

## 依赖

关键包（完整列表见 pyproject.toml）：
- `faster-whisper` - 语音识别
- `torch==2.5.1+cu121` - PyTorch（CUDA 12.1）
- `requests` - 翻译 API 的 HTTP 客户端
- `onnxruntime`、`ctranslate2` - 推理加速
- `pandas`、`tqdm` - 数据处理和进度条

系统要求：
- FFmpeg（必须在 PATH 中）
- 推荐 NVIDIA GPU（CUDA 可选，自动回退到 CPU）

## 重要说明

- 项目没有自动化测试；通过实际视频文件手动测试
- 日志同时输出到控制台和 `video_subtitle.log`
- 缓存存储在 `temp/translation_cache.db`（手动删除前永久有效）
- Whisper 模型首次使用时自动下载到 `temp/models/`
- FFmpeg 命令使用路径转义以确保字幕滤镜兼容性
- 文件操作显式使用 UTF-8 编码

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| CUDA 内存不足 | `uv run python video_subtitle.py input.mp4 --audio-segment 15` 或 `--model small` |
| Tokens 超限 | `uv run python video_subtitle.py input.mp4 --batch-size 20` |
| CPU 模式 | 自动切换 `COMPUTE_TYPE=int8` 和 `VIDEO_CODEC=libx264` |

## 另请参阅

- `AGENTS.md` - 更详细的开发指南，包含代码风格和安全注意事项
- `README.md` - 面向用户的文档，包含完整的配置示例
