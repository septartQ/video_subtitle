# 视频字幕生成与翻译工具 - Agent 指南

## 项目概述

这是一个基于 Python 的视频自动字幕生成与翻译工具，能够将视频中的语音转换为字幕，翻译成中文，并嵌入到视频中。项目采用单文件架构，所有核心功能集中在 `video_subtitle.py` 中。
### 回答规范

你回答用户的问题时必须使用中文，而非其它语言，仅在生产代码时使用英文

### 核心功能

1. **语音识别**：使用 faster-whisper 进行高精度语音识别，支持 CUDA 加速
2. **长视频分段处理**：超长音频自动分段识别，避免 CUDA OOM（显存溢出）
3. **批量翻译**：每 30 行字幕一次请求，避免 tokens 过多
4. **翻译缓存**：SQLite 缓存机制，相同内容只翻译一次
5. **字幕嵌入**：使用 FFmpeg 将字幕硬编码到视频中

### 技术栈

- **语言**：Python 3.8+
- **语音识别**：faster-whisper (OpenAI Whisper 的高效实现)
- **翻译 API**：阿里云百炼 (Bailian)
- **音视频处理**：FFmpeg
- **缓存**：SQLite3
- **依赖管理**：pip + requirements.txt

## 项目结构

```
project1/
├── video_subtitle.py      # 主程序文件，包含所有核心逻辑
├── requirements.txt       # Python 依赖
├── README.md              # 项目使用文档（中文）
├── FAQ.md                 # 常见问题（Kimi Code CLI 相关）
├── MCP.md                 # MCP 协议文档（Kimi Code CLI 相关）
├── 斜杠命令.md             # 斜杠命令文档（Kimi Code CLI 相关）
├── video_subtitle.log     # 运行日志
├── temp/                  # 临时文件目录
│   ├── translation_cache.db    # 翻译缓存数据库
│   ├── models/                 # Whisper 模型下载目录
│   ├── *_original.srt          # 原始语言字幕
│   ├── *_translated.srt        # 翻译后的中文字幕
│   └── *_audio_seg*.wav        # 分段音频文件（临时）
├── out/                   # 输出目录（可配置）
└── __pycache__/           # Python 字节码缓存
```

## 核心类架构

### Config (配置类)

位于文件顶部的 `@dataclass` 装饰的类，集中管理所有配置：

```python
@dataclass
class Config:
    BAILIAN_API_KEY: str          # 阿里云百炼 API 密钥
    BAILIAN_MODEL: str            # 翻译模型，默认 "qwen-mt-flash"
    TRANSLATION_BATCH_SIZE: int   # 每批翻译行数，默认 30
    WHISPER_MODEL: str            # Whisper 模型，默认 "large-v3"
    DEVICE: str                   # 计算设备，默认 "cuda"
    COMPUTE_TYPE: str             # 计算类型，默认 "float16"
    AUDIO_SEGMENT_MINUTES: int    # 音频分段长度，默认 30 分钟
    ENABLE_CACHE: bool            # 是否启用缓存，默认 True
```

### TranslationCache (翻译缓存)

基于 SQLite 的线程安全缓存实现：
- 使用 MD5 hash 作为缓存键
- 按模型名称区分缓存
- 支持线程本地连接
- 表结构：`text_hash`, `source_text`, `translated_text`, `model`, `created_at`

### AudioExtractor (音频提取器)

支持分段音频提取：
- 短视频直接提取为单文件
- 长视频自动分段（按 `AUDIO_SEGMENT_MINUTES` 配置）
- 输出 16kHz 单声道 WAV 格式

### SubtitleGenerator (字幕生成器)

Whisper 模型封装：
- 延迟加载模型（首次使用时加载）
- 支持 VAD（语音活动检测）
- 多段音频时间戳自动合并
- 输出 SRT 格式字幕

### SubtitleTranslator (字幕翻译器)

批量翻译实现：
- 支持缓存查询和写入
- 批量翻译（默认 30 行/次）
- 行数不匹配时自动回退到逐条翻译
- API 频率限制保护

### VideoEmbedder (视频嵌入器)

FFmpeg 字幕嵌入：
- 使用 `subtitles` 滤镜硬编码字幕
- 支持 NVENC 硬件编码（h264_nvenc）
- 可配置字幕样式（字体、大小、颜色等）

### VideoSubtitlePipeline (主流程)

 Orchestrates 完整处理流程：
1. 提取音频（支持分段）
2. 生成字幕（Whisper 识别）
3. 翻译字幕（批量+缓存）
4. 嵌入视频（FFmpeg 编码）

## 运行命令

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- `faster-whisper>=1.0.0` - 语音识别
- `requests>=2.31.0` - HTTP 请求
- `tqdm>=4.66.0` - 进度条（可选）

### 系统要求

- **FFmpeg**：必须安装并添加到 PATH
  - Windows: `choco install ffmpeg` 或 `scoop install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU**（推荐）：用于 CUDA 加速
- **Python**：3.8 或更高版本

### 运行程序

```bash
# 完整流程
python video_subtitle.py input.mp4

# 指定输出路径
python video_subtitle.py input.mp4 -o output.mp4

# 使用较小的模型（速度更快，精度较低）
python video_subtitle.py input.mp4 --model base

# 指定源语言（如英语）
python video_subtitle.py input.mp4 --language en

# 调整音频分段长度（解决显存不足）
python video_subtitle.py input.mp4 --audio-segment 15

# 调整翻译批量大小（解决 tokens 超限）
python video_subtitle.py input.mp4 --batch-size 20

# 跳过某些步骤
python video_subtitle.py input.mp4 --skip-translation  # 只生成原文字幕
python video_subtitle.py input.mp4 --skip-embedding    # 不嵌入视频

# 禁用翻译缓存
python video_subtitle.py input.mp4 --no-cache
```

### 可用模型

Whisper 模型大小选项：
- `tiny` - 最快，精度最低
- `base` - 较快
- `small` - 平衡
- `medium` - 较好
- `large-v1/v2/v3` - 最佳精度（默认 v3）

## 配置说明

### API 密钥配置

**必须修改** `video_subtitle.py` 中的以下配置：

```python
@dataclass
class Config:
    BAILIAN_API_KEY: str = "your-api-key-here"  # 替换为实际密钥
```

获取 API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

### 性能调优

**显存不足时**：
- 减小 `--audio-segment`（建议 15-20）
- 使用更小模型（如 `small` 或 `base`）
- 修改 `COMPUTE_TYPE = "int8"`
- 使用 `--device cpu`

**翻译质量优化**：
- 使用更好的模型：`BAILIAN_MODEL = "qwen-max"`
- 修改 `TRANSLATION_PROMPT` 添加领域特定要求

## 代码风格指南

### 命名约定

- **类名**：PascalCase（如 `TranslationCache`）
- **函数/方法**：snake_case（如 `translate_batch`）
- **常量**：UPPER_SNAKE_CASE（如 `TRANSLATION_BATCH_SIZE`）
- **私有方法**：单下划线前缀（如 `_rate_limit`）

### 类型注解

项目使用 Python 类型注解：
```python
def translate_batch(self, entries: List[dict]) -> List[str]:
    ...
```

### 日志记录

使用标准库 `logging`：
```python
logger.info(f"处理完成: {file_path}")
logger.warning(f"缓存查询失败: {e}")
logger.error(f"处理失败: {e}")
```

日志同时输出到控制台和 `video_subtitle.log` 文件。

### 异常处理

关键操作应有异常处理：
```python
try:
    result = subprocess.run(cmd, ...)
    if result.returncode != 0:
        raise RuntimeError(f"操作失败: {result.stderr}")
except Exception as e:
    logger.error(f"错误: {e}")
    raise
```

## 输出文件

处理完成后生成以下文件：

```
temp/
├── translation_cache.db              # 翻译缓存（持久化）
├── {video_name}_original.srt         # 原始语言字幕
└── {video_name}_translated.srt       # 翻译后的中文字幕

{video_dir}/
└── {video_name}_subtitled.mp4        # 最终带字幕视频
```

## 缓存管理

缓存数据库位于 `temp/translation_cache.db`：
- 以原文 MD5 hash 作为键
- 按模型名称区分缓存
- 缓存永久有效（手动删除数据库文件可清空）

清空缓存：
```bash
rm temp/translation_cache.db
```

## 测试策略

项目目前**没有自动化测试**，测试依赖：

1. **手动测试**：运行实际视频文件验证流程
2. **日志检查**：检查 `video_subtitle.log` 中的错误信息
3. **分段验证**：
   - 小文件测试（< 5分钟）
   - 中等文件测试（10-30分钟）
   - 大文件测试（> 1小时）

## 安全注意事项

1. **API 密钥**：`BAILIAN_API_KEY` 不应提交到版本控制
2. **文件路径**：使用 `os.path.abspath` 和 `Path` 处理路径
3. **命令注入**：FFmpeg 命令参数已转义处理
4. **线程安全**：SQLite 连接使用 `threading.local()`

## 故障排查

### CUDA 内存不足
```python
# 减小分段长度
config.AUDIO_SEGMENT_MINUTES = 15
# 使用更小模型
config.WHISPER_MODEL = "small"
# 降低精度
config.COMPUTE_TYPE = "int8"
```

### 翻译行数不匹配
系统会自动回退到逐条翻译模式，查看日志中的警告信息。

### FFmpeg 未找到
确保 FFmpeg 已安装并添加到系统 PATH，或修改代码使用完整路径。

## 开发注意事项

1. **单文件架构**：所有功能在一个文件中，新增功能应遵循现有类结构
2. **依赖延迟加载**：如 `faster_whisper` 和 `requests` 在使用时导入，非模块级别
3. **临时文件清理**：音频临时文件自动清理，字幕文件保留供用户查看
4. **模型下载**：Whisper 模型首次使用自动下载到 `temp/models/`
