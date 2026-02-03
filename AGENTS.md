# 视频字幕生成与翻译工具 - Agent 指南

## 项目概述

这是一个基于 Python 的视频自动字幕生成与翻译工具，能够将视频中的语音转换为字幕，翻译成中文，并嵌入到视频中。项目采用单文件架构，所有核心功能集中在 `video_subtitle.py` 中。

### 回答规范

你回答用户的问题时必须使用中文，而非其它语言，仅在生产代码时使用英文。

### 核心功能

1. **语音识别**：使用 faster-whisper 进行高精度语音识别，支持 CUDA 加速
2. **长视频分段处理**：超长音频自动分段识别，避免 CUDA OOM（显存溢出）
3. **批量翻译**：每 30 行字幕一次请求，避免 tokens 过多
4. **翻译缓存**：SQLite 缓存机制，相同内容只翻译一次
5. **字幕嵌入**：使用 FFmpeg 将字幕硬编码到视频中

### 技术栈

- **语言**：Python 3.12+
- **包管理器**：uv（优先）或 pip
- **语音识别**：faster-whisper (OpenAI Whisper 的高效实现)
- **翻译 API**：阿里云百炼 (Bailian/DashScope)
- **音视频处理**：FFmpeg
- **缓存**：SQLite3
- **依赖管理**：pyproject.toml + requirements.txt

## 项目结构

```
project1/
├── video_subtitle.py      # 主程序文件，包含所有核心逻辑（约 978 行）
├── pyproject.toml         # Python 项目配置（uv/pip 安装入口）
├── requirements.txt       # Python 依赖列表
├── uv.lock               # uv 包管理器锁定文件
├── .env.example          # 环境变量示例文件
├── .env                  # 实际环境变量（已加入 .gitignore）
├── .python-version       # Python 版本指定（3.12）
├── README.md             # 项目使用文档（中文）
├── LICENSE               # MIT 许可证
├── AGENTS.md             # 本文件
├── video_subtitle.log    # 运行日志
├── temp/                 # 临时文件目录
│   ├── translation_cache.db    # 翻译缓存数据库（SQLite）
│   ├── models/                 # Whisper 模型下载目录
│   ├── *_original.srt          # 原始语言字幕
│   ├── *_translated.srt        # 翻译后的中文字幕
│   └── *_audio_seg*.wav        # 分段音频文件（临时）
├── .venv/                # 虚拟环境（已加入 .gitignore）
├── __pycache__/          # Python 字节码缓存
└── *.egg-info/           # 包元数据目录
```

## 依赖管理

项目使用双重依赖管理方式：

### 方式一：使用 uv（推荐）

项目配置了 `pyproject.toml` 和 `uv.lock`，支持 uv 包管理器：

```bash
# 安装依赖（从 pyproject.toml）
uv pip install -e .

# 或从 requirements.txt
uv pip install -r requirements.txt

# 锁定依赖版本
uv lock
```

**Python 版本要求**：3.12（见 `.python-version` 和 `pyproject.toml` 中的 `requires-python`）

### 方式二：使用 pip

```bash
pip install -r requirements.txt
```

### 关键依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| faster-whisper | 1.2.1 | 语音识别 |
| torch | 2.5.1+cu121 | PyTorch (CUDA 12.1) |
| requests | 2.32.5 | HTTP 请求（翻译 API） |
| onnxruntime | 1.23.2 | ONNX 运行时 |
| ctranslate2 | 4.6.3 | 翻译加速 |
| pandas | 3.0.0 | 数据处理 |
| tqdm | 4.67.1 | 进度条 |

## 环境配置

### 必需：配置阿里云百炼 API Key

**方式一：使用 .env 文件（推荐）**

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件：
```bash
DASHSCOPE_API_KEY=sk-your-actual-api-key-here
```

**方式二：使用环境变量**

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-your-api-key"

# Linux/Mac
export DASHSCOPE_API_KEY="sk-your-api-key"
```

> ⚠️ **安全注意**：`.env` 文件已加入 `.gitignore`，不会被提交到 Git。

获取 API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

### 可选：CUDA 配置

项目默认使用 CUDA 加速（`DEVICE = "cuda"`）。如果 CUDA 不可用，程序会自动降级到 CPU 模式。

## 构建与运行命令

### 系统要求

- **FFmpeg**：必须安装并添加到 PATH
  - Windows: `choco install ffmpeg` 或 `scoop install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU**（推荐）：用于 CUDA 加速
- **Python**：3.12（由 `.python-version` 指定）

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

# 调整音频分段长度（解决显存不足，默认 30 分钟）
python video_subtitle.py input.mp4 --audio-segment 15

# 调整翻译批量大小（解决 tokens 超限，默认 30 行）
python video_subtitle.py input.mp4 --batch-size 20

# 跳过某些步骤
python video_subtitle.py input.mp4 --skip-translation  # 只生成原文字幕
python video_subtitle.py input.mp4 --skip-embedding    # 不嵌入视频
python video_subtitle.py input.mp4 --skip-transcription # 使用已有字幕

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

## 代码架构

### 核心类架构

```
Config (数据类)
├── 翻译配置（API Key、模型、批量大小）
├── Whisper 配置（模型、设备、计算类型）
├── 音频分段配置
├── 缓存配置
└── 字幕/视频编码配置

TranslationCache (翻译缓存管理器)
├── SQLite 数据库操作
├── MD5 hash 键值存储
└── 线程安全连接管理

AudioExtractor (音频提取器)
├── 短视频直接提取
└── 长视频分段提取（按 AUDIO_SEGMENT_MINUTES）

SubtitleGenerator (字幕生成器)
├── Whisper 模型延迟加载
├── VAD 语音活动检测
├── 多段音频时间戳合并
└── SRT 格式输出

SubtitleTranslator (字幕翻译器)
├── 缓存查询和写入
├── 批量翻译（默认 30 行/次）
├── 行数不匹配时逐条翻译回退
└── API 频率限制保护

VideoEmbedder (视频嵌入器)
├── FFmpeg 字幕硬编码
├── NVENC 硬件编码支持
└── 可配置字幕样式

VideoSubtitlePipeline (主流程编排)
├── 提取音频
├── 生成字幕
├── 翻译字幕
└── 嵌入视频
```

### 关键配置项（Config 类）

```python
@dataclass
class Config:
    # 阿里云百炼
    BAILIAN_API_KEY: str          # 从 DASHSCOPE_API_KEY 环境变量读取
    BAILIAN_MODEL: str = "qwen-mt-flash"
    API_RATE_LIMIT: float = 0.2   # 请求间隔（秒）
    
    # 翻译批量
    TRANSLATION_BATCH_SIZE: int = 30
    
    # Whisper
    WHISPER_MODEL: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    USE_VAD: bool = True
    
    # 音频分段处理
    AUDIO_SEGMENT_MINUTES: int = 30
    
    # 缓存
    ENABLE_CACHE: bool = True
    CACHE_DB_PATH: str = "./temp/translation_cache.db"
    
    # 字幕样式
    SUBTITLE_STYLE: str = "FontName=微软雅黑,..."
    SUBTITLE_MARGIN_V: int = 30
    
    # 视频编码
    VIDEO_CODEC: str = "h264_nvenc"  # 或 libx264
    VIDEO_CRF: int = 23
    AUDIO_CODEC: str = "aac"
    AUDIO_BITRATE: str = "192k"
```

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

使用标准库 `logging`，配置在模块级别：
```python
logger.info(f"处理完成: {file_path}")
logger.warning(f"缓存查询失败: {e}")
logger.error(f"处理失败: {e}")
```

日志同时输出到：
- 控制台（StreamHandler）
- `video_subtitle.log` 文件（FileHandler，UTF-8 编码）

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

### 环境变量加载

使用 `python-dotenv` 加载 `.env` 文件（如果安装）：
```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```

## 测试策略

项目目前**没有自动化测试**，测试依赖：

1. **手动测试**：运行实际视频文件验证流程
2. **日志检查**：检查 `video_subtitle.log` 中的错误信息
3. **分段验证**：
   - 小文件测试（< 5分钟）
   - 中等文件测试（10-30分钟）
   - 大文件测试（> 1小时）

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

**表结构**：
```sql
CREATE TABLE translations (
    text_hash TEXT PRIMARY KEY,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

清空缓存：
```bash
rm temp/translation_cache.db
```

## 安全注意事项

1. **API 密钥**：
   - `DASHSCOPE_API_KEY` 存储在 `.env` 文件，已加入 `.gitignore`
   - 代码中通过 `os.getenv("DASHSCOPE_API_KEY")` 读取
   - 绝不要硬编码真实 API Key 到代码中

2. **文件路径**：
   - 使用 `os.path.abspath` 转换为绝对路径
   - 使用 `pathlib.Path` 处理路径操作

3. **命令注入**：
   - FFmpeg 命令参数已转义处理（`srt_path.replace('\\', '/').replace(':', '\\:')`）
   - 使用列表形式传递命令参数，避免 shell 注入

4. **线程安全**：
   - SQLite 连接使用 `threading.local()` 实现线程本地存储

## 故障排查

### CUDA 内存不足
```bash
# 减小分段长度
python video_subtitle.py input.mp4 --audio-segment 15

# 使用更小模型
python video_subtitle.py input.mp4 --model small

# 切换到 CPU
python video_subtitle.py input.mp4 --device cpu
```

代码中也会自动检测 CUDA 可用性并降级：
```python
if config.DEVICE == 'cuda':
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，自动切换到 CPU 模式")
        config.DEVICE = 'cpu'
        config.COMPUTE_TYPE = 'int8'
        config.VIDEO_CODEC = 'libx264'
```

### 翻译行数不匹配
系统会自动回退到逐条翻译模式，查看日志中的警告信息。

### FFmpeg 未找到
确保 FFmpeg 已安装并添加到系统 PATH，或修改代码使用完整路径。

## 开发注意事项

1. **单文件架构**：所有功能在一个文件中，新增功能应遵循现有类结构

2. **依赖延迟加载**：如 `faster_whisper` 和 `requests` 在使用时导入，非模块级别
   ```python
   def load_model(self):
       from faster_whisper import WhisperModel
       ...
   ```

3. **临时文件清理**：
   - 音频临时文件在字幕生成后自动清理
   - 字幕文件保留供用户查看
   - 可通过 `config.KEEP_TEMP = True` 保留临时文件

4. **模型下载**：
   - Whisper 模型首次使用自动下载到 `temp/models/`
   - 下载进度会显示在控制台

5. **API 响应格式**：
   - 阿里云百炼 API 支持两种响应格式：
     - `result["output"]["choices"][0]["message"]["content"]`
     - `result["output"]["text"]`
