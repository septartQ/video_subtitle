# 视频自动字幕生成与翻译工具

一个完整的视频字幕处理工具，支持**长视频分段处理**、**多平台翻译**和**翻译缓存**。

## 功能特点

1. **语音识别**：使用 faster-whisper 进行高精度语音识别，支持 CUDA 加速
2. **长视频分段**：超长音频自动分段识别，避免 CUDA OOM
3. **多平台翻译**：支持阿里云百炼、硅基流动、DeepSeek 等翻译平台
4. **批量翻译**：每 30 行字幕一次请求，避免 tokens 过多
5. **翻译缓存**：SQLite 缓存，相同内容只翻译一次（按模型隔离）
6. **字幕嵌入**：使用 FFmpeg 将字幕硬编码到视频中

## 环境要求

- Python 3.12（推荐，项目指定版本）
- uv 或 pip（包管理器）
- NVIDIA GPU（推荐，用于 CUDA 加速）
- FFmpeg

### 安装 uv（可选但推荐）

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

验证安装：
```bash
uv --version
```

## 快速开始

### 1. 安装依赖

本项目支持使用 **uv**（推荐）或 **pip** 管理依赖。

#### 方式一：使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是一个极速的 Python 包管理器，比 pip 快 10-100 倍。

```bash
# 创建虚拟环境（自动使用项目指定的 Python 版本）
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖（二选一）
uv pip install -e .           # 从 pyproject.toml 安装（推荐）
# 或
uv pip install -r requirements.txt  # 从 requirements.txt 安装

# 或使用 uv run 直接运行（无需手动激活虚拟环境）
uv run python video_subtitle.py input.mp4
```

#### 方式二：使用 pip

```bash
# 创建虚拟环境（可选但推荐）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```


### 2. 安装 FFmpeg

**Windows（推荐方式）**:
```powershell
# 使用 winget（Windows 自带，无需额外安装）
winget install Gyan.FFmpeg

# 或使用 Chocolatey
choco install ffmpeg

# 或使用 Scoop
scoop install ffmpeg
```

**macOS**:
```bash
brew install ffmpeg
```

**Linux**:
```bash
sudo apt update && sudo apt install ffmpeg
```

### 3. 配置翻译平台 API

选择以下任一平台，获取 API Key 并填入 `.env` 文件。

#### 阿里云百炼（默认）

获取 API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

```bash
# .env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

使用：
```bash
python video_subtitle.py input.mp4                    # 默认
python video_subtitle.py input.mp4 --provider bailian
```

#### 硅基流动

获取 API Key: https://cloud.siliconflow.cn/account/ak

```bash
# .env
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
# 可选：覆盖默认模型
# SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
```

使用：
```bash
python video_subtitle.py input.mp4 --provider siliconflow
```

#### DeepSeek

获取 API Key: https://platform.deepseek.com/api_keys

```bash
# .env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
# 可选：覆盖默认模型
# DEEPSEEK_MODEL=deepseek-chat
```

使用：
```bash
python video_subtitle.py input.mp4 --provider deepseek
```

#### 配置方式

**方式一：使用 .env 文件（推荐）**

```bash
cp .env.example .env
# 编辑 .env 填入你选择平台的 API Key
```

**方式二：使用环境变量**

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Linux/Mac
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

> ⚠️ **注意**：`.env` 文件已添加到 `.gitignore`，不会被提交到 Git。

### 4. 测试 API 连通性（可选）

```bash
python video_subtitle.py --test-api                          # 测试默认平台（百炼）
python video_subtitle.py --test-api --provider siliconflow   # 测试硅基流动
python video_subtitle.py --test-api --provider deepseek      # 测试 DeepSeek
```

输出示例：
```
✅ API 连通性测试通过
```

### 5. 运行程序

#### 使用 uv（推荐）

```bash
# 完整流程
uv run python video_subtitle.py input.mp4

# 使用硅基流动翻译
uv run python video_subtitle.py input.mp4 --provider siliconflow

# 使用 DeepSeek 翻译
uv run python video_subtitle.py input.mp4 --provider deepseek

# 指定输出路径
uv run python video_subtitle.py input.mp4 -o output.mp4

# 使用较小的模型（速度更快）
uv run python video_subtitle.py input.mp4 --model base

# 指定源语言
uv run python video_subtitle.py input.mp4 --language en

# 测试 API 连通性
uv run python video_subtitle.py --test-api
```

#### 使用 pip（需先激活虚拟环境）

```bash
# 完整流程
python video_subtitle.py input.mp4

# 使用硅基流动翻译
python video_subtitle.py input.mp4 --provider siliconflow

# 使用 DeepSeek 翻译
python video_subtitle.py input.mp4 --provider deepseek

# 指定输出路径
python video_subtitle.py input.mp4 -o output.mp4

# 使用较小的模型（速度更快）
python video_subtitle.py input.mp4 --model base

# 指定源语言
python video_subtitle.py input.mp4 --language en

# 测试 API 连通性
python video_subtitle.py --test-api
```

## 分段处理配置

### 音频分段（语音识别）

```bash
# uv
uv run python video_subtitle.py input.mp4 --audio-segment 20

# pip
python video_subtitle.py input.mp4 --audio-segment 20
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--audio-segment` | 音频分段长度（分钟） | 30 |

**何时需要调整？**
- 显存较小（< 8GB）：设为 15-20 分钟
- 显存充足（> 12GB）：可保持 30 分钟或更大

## 翻译配置

### 翻译平台

```bash
python video_subtitle.py input.mp4 --provider siliconflow
```

| 平台 | `--provider` 值 | 默认模型 | 环境变量 |
|------|----------------|---------|----------|
| 阿里云百炼 | `bailian`（默认） | `qwen-mt-flash` | `DASHSCOPE_API_KEY` |
| 硅基流动 | `siliconflow` | `Qwen/Qwen2.5-7B-Instruct` | `SILICONFLOW_API_KEY` |
| DeepSeek | `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |

也可通过环境变量设置：
```bash
# .env
TRANSLATION_PROVIDER=deepseek
```

### 批量大小

```bash
# uv
uv run python video_subtitle.py input.mp4 --batch-size 50

# pip
python video_subtitle.py input.mp4 --batch-size 50
```

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `--batch-size` | 每批翻译行数 | 30 | 20-100 |

**调整建议：**
- 字幕较短（单词少）：可增大到 50-100
- 字幕较长（句子长）：减小到 20-30，避免 tokens 超限
- 出现 API 错误（tokens 不足）：减小批量大小

### 翻译缓存

缓存默认启用，存储在 `temp/translation_cache.db`。

```bash
# 禁用缓存（uv）
uv run python video_subtitle.py input.mp4 --no-cache

# 禁用缓存（pip）
python video_subtitle.py input.mp4 --no-cache
```

**缓存机制：**
- 以原文 MD5 hash 作为键
- 按模型名称区分缓存
- 缓存永久有效（手动删除数据库文件可清空）

**适用场景：**
- 系列课程（大量重复术语）
- 重复处理相同视频（跳过已翻译内容）
- 网络不稳定时（减少 API 调用次数）

## 完整配置示例

```python
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """配置类"""
    # === 翻译平台配置 ===
    TRANSLATION_PROVIDER: str = "bailian"  # bailian / siliconflow / deepseek
    
    # === 阿里云百炼配置 ===
    BAILIAN_API_KEY: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE"))
    BAILIAN_MODEL: str = "qwen-mt-flash"
    
    # === 硅基流动配置 ===
    SILICONFLOW_API_KEY: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "YOUR_API_KEY_HERE"))
    SILICONFLOW_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    
    # === DeepSeek 配置 ===
    DEEPSEEK_API_KEY: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"))
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    
    API_RATE_LIMIT: float = 0.2  # 请求间隔（秒）
    
    # === 翻译切片配置 ===
    TRANSLATION_BATCH_SIZE: int = 30  # 每批翻译行数
    
    # === Whisper 配置 ===
    WHISPER_MODEL: str = "large-v3"  # 可选: tiny, base, small, medium, large-v1/v2/v3
    DEVICE: str = "cuda"  # 或 cpu
    COMPUTE_TYPE: str = "float16"  # CPU 模式自动切换为 int8
    USE_VAD: bool = True  # 启用语音活动检测
    VAD_PARAMETERS: dict = field(default_factory=lambda: {
        "min_silence_duration_ms": 500,
        "max_speech_duration_s": 30,
    })
    SOURCE_LANGUAGE: Optional[str] = None  # 源语言代码，None 为自动检测
    
    # === 音频分段处理配置 ===
    AUDIO_SEGMENT_MINUTES: int = 30  # 超过此长度分段处理
    
    # === 翻译缓存配置 ===
    ENABLE_CACHE: bool = True
    CACHE_DB_PATH: str = "./temp/translation_cache.db"
    
    # === 字幕配置 ===
    SUBTITLE_STYLE: str = (
        "FontName=微软雅黑,"
        "FontSize=24,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2"
    )
    SUBTITLE_MARGIN_V: int = 30  # 字幕底部边距
    
    # === 输出配置 ===
    VIDEO_CODEC: str = "h264_nvenc"  # 或 libx264（CPU 模式自动切换）
    VIDEO_CRF: int = 23  # 质量参数（越小质量越高，文件越大）
    AUDIO_CODEC: str = "aac"
    AUDIO_BITRATE: str = "192k"
    TEMP_DIR: str = "./temp"  # 临时文件目录
    KEEP_TEMP: bool = False  # 是否保留临时文件
```

## 输出文件

处理完成后，文件保存在：

```
temp/
├── translation_cache.db              # 翻译缓存数据库
├── {video_name}_original.srt         # 原始语言字幕
└── {video_name}_translated.srt       # 翻译后的中文字幕

{video_dir}/
└── {video_name}_subtitled.mp4        # 最终带字幕视频
```

## 常见问题

### Q: 如何检查 CUDA 是否可用？

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Q: CUDA 内存不足怎么办？

1. 减小音频分段长度：`--audio-segment 15`
2. 使用更小的模型：`--model small`
3. 使用 int8 精度：修改 `COMPUTE_TYPE = "int8"`
4. 切换到 CPU：`--device cpu`

### Q: 翻译出现 "tokens 超限" 错误？

减小批量大小：`--batch-size 20` 或更小

### Q: 如何提高翻译质量？

- 尝试不同的翻译平台：`--provider deepseek` 或 `--provider siliconflow`
- 使用更好的模型：修改对应平台的 MODEL 环境变量
- 修改 `TRANSLATION_PROMPT` 添加领域特定要求

### Q: 缓存占用空间太大？

直接删除缓存文件：
```bash
rm temp/translation_cache.db
```

## 处理流程图

```
输入视频
    │
    ├─► 提取音频 ──► [分段？] ──► 分段提取 ──┐
    │                                    │
    └────────────────────────────────────┘
                    │
                    ▼
            Whisper 语音识别
            （分段识别，合并时间戳）
                    │
                    ▼
            生成 original.srt
                    │
                    ▼
            批量翻译（切片）
            ┌─ 检查缓存
            ├─ API 调用
            └─ 存入缓存
                    │
                    ▼
            生成 translated.srt
                    │
                    ▼
                    │
                    ▼
            嵌入视频（FFmpeg 硬编码字幕）
                    │
                    ▼
            输出 subtitled.mp4
```

## 许可证

MIT License