# 视频自动字幕生成与翻译工具

一个完整的视频字幕处理工具，支持**长视频分段处理**、**批量翻译**和**翻译缓存**。

## 功能特点

1. **语音识别**：使用 faster-whisper 进行高精度语音识别，支持 CUDA 加速
2. **长视频分段**：超长音频自动分段识别，避免 CUDA OOM
3. **批量翻译**：每 30 行字幕一次请求，避免 tokens 过多
4. **翻译缓存**：SQLite 缓存，相同内容只翻译一次
5. **字幕嵌入**：使用 FFmpeg 将字幕硬编码到视频中

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

### 3. 配置阿里云百炼 API

**方式一：使用 .env 文件（推荐）**

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API Key：
```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**方式二：使用环境变量**

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Linux/Mac
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

获取 API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

> ⚠️ **注意**：`.env` 文件已添加到 `.gitignore`，不会被提交到 Git，请放心使用。

### 4. 测试 API 连通性（可选）

在正式处理视频前，建议先测试 API 是否配置正确：

```bash
python video_subtitle.py --test-api
```

输出示例：
```
✅ API 连通性测试通过
```

### 5. 运行程序

#### 使用 uv（推荐）

```bash
# 完整流程（自动使用虚拟环境中的依赖）
uv run python video_subtitle.py input.mp4

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
@dataclass
class Config:
    # 阿里云百炼
    BAILIAN_API_KEY: str = "your-api-key"
    BAILIAN_MODEL: str = "qwen-turbo"  # 或 qwen-plus, qwen-max
    API_RATE_LIMIT: float = 0.2
    
    # 翻译批量
    TRANSLATION_BATCH_SIZE: int = 30
    
    # Whisper
    WHISPER_MODEL: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    
    # 音频分段处理
    AUDIO_SEGMENT_MINUTES: int = 30    # 音频分段
    
    # 缓存
    ENABLE_CACHE: bool = True
    CACHE_DB_PATH: str = "./temp/translation_cache.db"
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

- 使用更好的模型：`BAILIAN_MODEL = "qwen-max"`
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
