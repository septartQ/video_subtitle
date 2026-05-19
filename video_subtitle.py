#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频自动字幕生成与翻译工具（支持长视频分段处理）
功能：
1. 从视频提取音频并生成SRT字幕（支持 CUDA 加速，长视频自动分段）
2. 将字幕批量翻译（切片处理，每30行一次请求）
3. 将翻译后的字幕硬嵌入视频
4. 翻译结果自动缓存（SQLite）

作者：AI Assistant
日期：2026-05-19
"""

import os
import sys
import re
import time
import hashlib
import shutil
import sqlite3
import subprocess
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_subtitle.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== 配置区域 ====================

@dataclass
class Config:
    """配置类"""
    # === 翻译平台配置 ===
    # 翻译平台: bailian / siliconflow / deepseek
    TRANSLATION_PROVIDER: str = field(default_factory=lambda: os.getenv("TRANSLATION_PROVIDER", "bailian"))

    # === 阿里云百炼配置 ===
    # 从环境变量读取，如未设置则使用占位符
    BAILIAN_API_KEY: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE"))
    BAILIAN_MODEL: str = "qwen-mt-flash"

    # === 硅基流动配置 ===
    SILICONFLOW_API_KEY: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", "YOUR_API_KEY_HERE"))
    SILICONFLOW_MODEL: str = field(default_factory=lambda: os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    # === DeepSeek 配置 ===
    DEEPSEEK_API_KEY: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"))
    DEEPSEEK_MODEL: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

    API_RATE_LIMIT: float = 0.2
    API_TIMEOUT: int = 120
    TRANSLATION_TEMPERATURE: float = 0.3
    
    # === 翻译切片配置 ===
    # 每批次翻译的字幕行数（避免 tokens 过多）
    TRANSLATION_BATCH_SIZE: int = 30
    
    # === Whisper 配置 ===
    WHISPER_MODEL: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    USE_VAD: bool = True
    VAD_PARAMETERS: dict = field(default_factory=lambda: {
        "min_silence_duration_ms": 500,
        "max_speech_duration_s": 15,
    })
    SOURCE_LANGUAGE: Optional[str] = None
    TARGET_LANGUAGE: str = "中文"

    # === 音频分段处理配置 ===
    # 音频分段长度（分钟），超过此长度的视频会分段处理
    AUDIO_SEGMENT_MINUTES: int = 30
    
    # === 翻译缓存配置 ===
    # 启用翻译缓存
    ENABLE_CACHE: bool = True
    
    # 缓存数据库路径
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
    SUBTITLE_MARGIN_V: int = 30
    
    # === 输出配置 ===
    VIDEO_CODEC: str = "h264_nvenc"
    VIDEO_CRF: int = 23
    AUDIO_CODEC: str = "aac"
    AUDIO_BITRATE: str = "192k"
    TEMP_DIR: str = "./temp"
    KEEP_TEMP: bool = False
    
    # 翻译提示词模板（批量翻译版本）
    TRANSLATION_PROMPT: str = """请将以下视频字幕翻译成{target_lang}。
要求：
1. 翻译要自然、通顺，符合{target_lang}的表达习惯
2. 保持字幕的时间戳不变
3. 如果原文已经是{target_lang}，请直接返回原文
4. 不要添加任何解释或额外内容
5. 确保翻译后的文本长度适合字幕显示
6. 必须按照原始格式返回，每行字幕一行翻译

输入格式：每行一个字幕条目，格式为 "编号|原文"
输出格式：只返回翻译后的文本，每行一个，与输入行数严格对应

原文：
{text}

请只返回翻译后的文本（每行一个）："""


# ==================== 翻译平台 Provider ====================

_MISSING = object()  # API Key 未设置占位符


class TranslationProvider(ABC):
    """翻译平台抽象基类"""

    @property
    @abstractmethod
    def api_key(self) -> str:
        pass

    @abstractmethod
    def get_model(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def _build_payload(self, messages: List[Dict], max_tokens: int, temperature: float) -> Dict:
        """构建请求体"""
        pass

    @abstractmethod
    def _parse_response(self, result: Dict) -> str:
        """从响应中提取翻译文本"""
        pass

    @abstractmethod
    def _get_endpoint(self) -> str:
        """获取 API 端点 URL"""
        pass

    def _api_post(self, payload: Dict, timeout: int):
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        return requests.post(self._get_endpoint(), headers=headers, json=payload, timeout=timeout)

    def chat(self, messages: List[Dict], max_tokens: int = 4000, temperature: float = 0.3) -> str:
        payload = self._build_payload(messages, max_tokens, temperature)
        response = self._api_post(payload, timeout=self.config.API_TIMEOUT)
        if response.status_code == 200:
            return self._parse_response(response.json())
        raise RuntimeError(f"API 错误: HTTP {response.status_code}")

    def test_connection(self) -> bool:
        import requests

        logger.info(f"正在测试 {self.get_name()} API 连通性...")

        if not self.api_key or self.api_key is _MISSING:
            logger.error(f"❌ {self.get_name()} API Key 未设置")
            return False

        try:
            payload = self._build_payload(
                [{"role": "user", "content": "Hello"}],
                max_tokens=10, temperature=0
            )
            response = self._api_post(payload, timeout=30)

            if response.status_code == 200:
                logger.info(f"✅ 连通性测试通过")
                return True
            elif response.status_code == 401:
                logger.error(f"❌ API Key 无效或已过期 (HTTP {response.status_code})")
                return False
            elif response.status_code == 429:
                logger.warning(f"⚠️ 请求过于频繁 (HTTP {response.status_code})")
                return False
            else:
                logger.error(f"❌ API 请求失败: HTTP {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error("❌ 请求超时，请检查网络连接")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("❌ 网络连接错误，请检查网络或代理设置")
            return False
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False


class BailianProvider(TranslationProvider):
    """阿里云百炼（DashScope）翻译 Provider"""

    def __init__(self, config: Config):
        self.config = config

    @property
    def api_key(self) -> str:
        return self.config.BAILIAN_API_KEY

    def get_model(self) -> str:
        return self.config.BAILIAN_MODEL

    def get_name(self) -> str:
        return "阿里云百炼"

    def _get_endpoint(self) -> str:
        return "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _build_payload(self, messages: List[Dict], max_tokens: int, temperature: float) -> Dict:
        return {
            "model": self.get_model(),
            "input": {"messages": messages},
            "parameters": {
                "result_format": "message",
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }

    def _parse_response(self, result: Dict) -> str:
        if "output" in result and "choices" in result["output"]:
            return result["output"]["choices"][0]["message"]["content"]
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"]
        raise RuntimeError("无法解析 API 响应")


class OpenAICompatProvider(TranslationProvider):
    """OpenAI 兼容 API Provider（硅基流动、DeepSeek 等）"""

    def __init__(self, config: Config, name: str, api_key: str, model: str, base_url: str):
        self.config = config
        self._name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip('/')

    @property
    def api_key(self) -> str:
        return self._api_key

    def get_model(self) -> str:
        return self._model

    def get_name(self) -> str:
        return self._name

    def _get_endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _build_payload(self, messages: List[Dict], max_tokens: int, temperature: float) -> Dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def _parse_response(self, result: Dict) -> str:
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise RuntimeError("无法解析 API 响应")


# ==================== 翻译缓存管理器 ====================

class TranslationCache:
    """翻译缓存管理器（SQLite 实现）"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        parent = Path(db_path).parent
        if str(parent) and not parent.exists():
            os.makedirs(parent, exist_ok=True)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                text_hash TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash ON translations(text_hash)
        ''')
        conn.commit()
        conn.close()
        logger.info(f"翻译缓存初始化完成: {self.db_path}")
    
    def _get_hash(self, text: str) -> str:
        """计算文本的 MD5 hash"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[str]:
        """获取缓存的翻译结果"""
        if not text.strip():
            return ""
        
        text_hash = self._get_hash(text)
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT translated_text FROM translations WHERE text_hash = ? AND model = ?",
                (text_hash, model)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"缓存查询失败: {e}", exc_info=True)
        return None
    
    def set(self, text: str, translated: str, model: str):
        """设置缓存"""
        if not text.strip():
            return
        
        text_hash = self._get_hash(text)
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO translations (text_hash, source_text, translated_text, model)
                VALUES (?, ?, ?, ?)
            ''', (text_hash, text, translated, model))
            conn.commit()
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}", exc_info=True)
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), model FROM translations GROUP BY model")
            stats = cursor.fetchall()
            return {"total_entries": sum(s[0] for s in stats), "by_model": stats}
        except Exception as e:
            return {"error": str(e)}


# ==================== 工具函数 ====================

def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否已安装"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def format_time(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_time(time_str: str) -> float:
    """将时间字符串转换为秒数，支持 HH:MM:SS,mmm 和 HH:MM:SS.xx 格式"""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) < 2 or len(parts) > 3:
        logger.warning(f"无效的时间格式: {time_str}")
        return 0.0
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except (ValueError, IndexError):
        logger.warning(f"无法解析时间: {time_str}")
        return 0.0


def parse_srt(srt_content: str) -> List[dict]:
    """解析 SRT 文件内容为结构化数据"""
    entries = []
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\n|\n\d+\s*\n|$)'
    matches = re.findall(pattern, srt_content, re.DOTALL)
    
    for match in matches:
        index, start, end, text = match
        entries.append({
            'index': int(index),
            'start': start,
            'end': end,
            'text': text.strip().replace('\n', ' ')
        })
    return entries


def write_srt(entries: List[dict], output_path: str):
    """将字幕条目写入 SRT 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(f"{entry['index']}\n")
            f.write(f"{entry['start']} --> {entry['end']}\n")
            f.write(f"{entry['text']}\n\n")


def shift_srt_time(entries: List[dict], offset_seconds: float) -> List[dict]:
    """调整字幕时间戳"""
    shifted = []
    for entry in entries:
        start_sec = parse_time(entry['start']) + offset_seconds
        end_sec = parse_time(entry['end']) + offset_seconds
        shifted.append({
            'index': entry['index'],
            'start': format_time(max(0, start_sec)),
            'end': format_time(max(0, end_sec)),
            'text': entry['text']
        })
    return shifted


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        logger.error(f"ffprobe 获取视频时长失败: {result.stderr.strip()}")
        raise RuntimeError(f"无法获取视频时长: {result.stderr.strip()}")
    try:
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise ValueError(f"无效的时长: {duration}")
        return duration
    except ValueError:
        logger.error(f"无法解析视频时长: {result.stdout.strip()}")
        raise RuntimeError(f"无法解析视频时长: {result.stdout.strip()}")


def get_video_size_gb(video_path: str) -> float:
    """获取视频大小（GB）"""
    return os.path.getsize(video_path) / (1024 ** 3)


# ==================== 核心类 ====================

class AudioExtractor:
    """音频提取器（支持分段）"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    def extract_segment(self, video_path: str, start: float, duration: float, output_path: str):
        """提取指定时间段的音频"""
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-ss', str(start), '-t', str(duration),
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(f"音频提取失败: {result.stderr}")
    
    def extract(self, video_path: str) -> List[str]:
        """提取音频，如果视频过长则分段"""
        duration = get_video_duration(video_path)
        segment_duration = self.config.AUDIO_SEGMENT_MINUTES * 60
        
        if duration <= segment_duration:
            video_name = Path(video_path).stem
            audio_path = os.path.join(self.config.TEMP_DIR, f"{video_name}_audio.wav")
            logger.info(f"正在提取音频: {video_path} (时长: {duration/60:.1f}分钟)")
            self.extract_segment(video_path, 0, duration, audio_path)
            logger.info(f"音频提取完成: {audio_path}")
            return [audio_path]
        else:
            num_segments = int(duration / segment_duration) + 1
            logger.info(f"视频时长 {duration/60:.1f}分钟，将分成 {num_segments} 段处理")
            
            audio_segments = []
            video_name = Path(video_path).stem
            
            for i in range(num_segments):
                start = i * segment_duration
                seg_duration = min(segment_duration, duration - start)
                output_path = os.path.join(
                    self.config.TEMP_DIR, 
                    f"{video_name}_audio_seg{i:03d}.wav"
                )
                
                logger.info(f"提取音频段 {i+1}/{num_segments}: {start/60:.1f}min - {(start+seg_duration)/60:.1f}min")
                try:
                    self.extract_segment(video_path, start, seg_duration, output_path)
                    audio_segments.append(output_path)
                except Exception as e:
                    logger.error(f"音频段 {i+1} 提取失败: {e}")
                    for path in audio_segments:
                        if os.path.exists(path):
                            os.remove(path)
                    raise RuntimeError(f"音频提取在第 {i+1}/{num_segments} 段失败") from e
            
            return audio_segments


class SubtitleGenerator:
    """字幕生成器（支持分段识别）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load_model(self):
        """加载 Whisper 模型"""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                
                logger.info(f"正在加载 Whisper 模型: {self.config.WHISPER_MODEL}")
                logger.info(f"设备: {self.config.DEVICE}, 计算类型: {self.config.COMPUTE_TYPE}")
                
                self.model = WhisperModel(
                    self.config.WHISPER_MODEL,
                    device=self.config.DEVICE,
                    compute_type=self.config.COMPUTE_TYPE,
                    download_root=os.path.join(self.config.TEMP_DIR, "models")
                )
                
                logger.info("模型加载完成")
                
            except ImportError:
                logger.error("请先安装 faster-whisper: pip install faster-whisper")
                raise
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
    
    def transcribe_audio(self, audio_path: str) -> List[dict]:
        """识别单个音频文件"""
        segments, info = self.model.transcribe(
            audio_path,
            language=self.config.SOURCE_LANGUAGE,
            vad_filter=self.config.USE_VAD,
            vad_parameters=self.config.VAD_PARAMETERS if self.config.USE_VAD else None,
            condition_on_previous_text=True,
            word_timestamps=False
        )
        
        entries = []
        for segment in segments:
            entries.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        return entries
    
    def generate(self, audio_segments: List[str], video_path: str) -> str:
        """生成 SRT 字幕文件（支持多段音频）"""
        self.load_model()
        
        video_name = Path(video_path).stem
        srt_path = os.path.join(self.config.TEMP_DIR, f"{video_name}_original.srt")
        
        segment_duration = self.config.AUDIO_SEGMENT_MINUTES * 60
        all_entries = []
        entry_index = 1
        
        for seg_idx, audio_path in enumerate(audio_segments):
            logger.info(f"识别音频段 {seg_idx+1}/{len(audio_segments)}: {audio_path}")
            
            time_offset = seg_idx * segment_duration
            entries = self.transcribe_audio(audio_path)
            
            # 调整时间戳
            for entry in entries:
                all_entries.append({
                    'index': entry_index,
                    'start': format_time(entry['start'] + time_offset),
                    'end': format_time(entry['end'] + time_offset),
                    'text': entry['text']
                })
                entry_index += 1
            
            logger.info(f"段 {seg_idx+1} 识别完成: {len(entries)} 条字幕")
        
        # 写入 SRT 文件
        write_srt(all_entries, srt_path)
        logger.info(f"字幕生成完成: {srt_path} (共 {len(all_entries)} 条)")
        return srt_path


class SubtitleTranslator:
    """字幕翻译器（支持批量切片、缓存和多平台）"""

    def __init__(self, config: Config):
        self.config = config
        self.last_request_time = 0
        self.cache = TranslationCache(config.CACHE_DB_PATH) if config.ENABLE_CACHE else None

        provider_name = config.TRANSLATION_PROVIDER.lower()
        if provider_name == "siliconflow":
            self.provider = OpenAICompatProvider(
                config, "硅基流动",
                config.SILICONFLOW_API_KEY,
                config.SILICONFLOW_MODEL,
                config.SILICONFLOW_BASE_URL
            )
        elif provider_name == "deepseek":
            self.provider = OpenAICompatProvider(
                config, "DeepSeek",
                config.DEEPSEEK_API_KEY,
                config.DEEPSEEK_MODEL,
                config.DEEPSEEK_BASE_URL
            )
        else:
            self.provider = BailianProvider(config)

        self.model = self.provider.get_model()

        if not self.provider.api_key or self.provider.api_key is _MISSING:
            logger.warning(f"警告: 请设置有效的 {self.provider.get_name()} API Key")

    def _rate_limit(self):
        """频率限制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.API_RATE_LIMIT:
            sleep_time = self.config.API_RATE_LIMIT - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def test_connection(self) -> bool:
        """测试API连通性"""
        self._rate_limit()
        return self.provider.test_connection()

    def translate_batch(self, entries: List[dict]) -> List[str]:
        """批量翻译字幕条目"""
        if not entries:
            return []

        original_entries = entries

        # 检查缓存
        if self.cache:
            cached_results = []
            need_translate = []
            for entry in entries:
                cached = self.cache.get(entry['text'], self.model)
                if cached:
                    cached_results.append((entry['index'], cached))
                else:
                    need_translate.append(entry)

            if cached_results:
                logger.info(f"缓存命中 {len(cached_results)}/{len(entries)} 条")

            if not need_translate:
                cached_results.sort(key=lambda x: x[0])
                return [text for _, text in cached_results]

            entries = need_translate

        batch_text = "\n".join([f"{e['index']}|{e['text']}" for e in entries])
        prompt = self.config.TRANSLATION_PROMPT.format(
            text=batch_text,
            target_lang=self.config.TARGET_LANGUAGE
        )

        self._rate_limit()

        try:
            translated_text = self.provider.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=self.config.TRANSLATION_TEMPERATURE
            )

            translated_lines = [line.strip() for line in translated_text.strip().split('\n') if line.strip()]

            if len(translated_lines) != len(entries):
                logger.warning(f"翻译行数不匹配: 输入{len(entries)}行，输出{len(translated_lines)}行，将使用逐条翻译")
                return self._translate_one_by_one(original_entries)

            if self.cache:
                for entry, translated in zip(entries, translated_lines):
                    self.cache.set(entry['text'], translated, self.model)

            if self.cache:
                all_results = cached_results + [(e['index'], t) for e, t in zip(entries, translated_lines)]
                all_results.sort(key=lambda x: x[0])
                return [text for _, text in all_results]

            return translated_lines

        except Exception as e:
            logger.error(f"批量翻译失败: {e}")
            return self._translate_one_by_one(original_entries)

    def _translate_one_by_one(self, entries: List[dict]) -> List[str]:
        """逐条翻译（备用方案）"""
        results = []
        for entry in entries:
            if self.cache:
                cached = self.cache.get(entry['text'], self.model)
                if cached:
                    results.append(cached)
                    continue

            translated = self._translate_single(entry['text'])
            if self.cache:
                self.cache.set(entry['text'], translated, self.model)
            results.append(translated)
        return results

    def _translate_single(self, text: str) -> str:
        """翻译单条文本"""
        if not text.strip():
            return ""

        self._rate_limit()

        try:
            prompt = f"将以下文本翻译成{self.config.TARGET_LANGUAGE}，只返回翻译结果:\n{text}"
            return self.provider.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=self.config.TRANSLATION_TEMPERATURE
            ).strip()
        except Exception as e:
            logger.error(f"翻译失败: {e}")
            return text

    def translate_srt(self, srt_path: str, video_path: str) -> str:
        """翻译整个 SRT 文件（切片处理）"""
        video_name = Path(video_path).stem
        translated_path = os.path.join(
            self.config.TEMP_DIR,
            f"{video_name}_translated.srt"
        )

        logger.info(f"开始翻译字幕: {srt_path}")
        logger.info(f"翻译平台: {self.provider.get_name()}")
        logger.info(f"使用模型: {self.model}")
        logger.info(f"批量大小: {self.config.TRANSLATION_BATCH_SIZE} 行/次")

        with open(srt_path, 'r', encoding='utf-8-sig') as f:
            srt_content = f.read()

        entries = parse_srt(srt_content)
        total = len(entries)
        logger.info(f"共 {total} 条字幕需要翻译")

        if self.cache:
            stats = self.cache.get_stats()
            logger.info(f"缓存统计: {stats.get('total_entries', 0)} 条历史记录")

        translated_entries = []
        batch_size = self.config.TRANSLATION_BATCH_SIZE

        for i in range(0, total, batch_size):
            batch = entries[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"翻译批次 {batch_num}/{total_batches} (条目 {i+1}-{min(i+batch_size, total)})")

            translated_texts = self.translate_batch(batch)

            for entry, translated in zip(batch, translated_texts):
                translated_entries.append({
                    'index': entry['index'],
                    'start': entry['start'],
                    'end': entry['end'],
                    'text': translated
                })

        write_srt(translated_entries, translated_path)

        if self.cache:
            stats = self.cache.get_stats()
            logger.info(f"翻译完成！缓存共 {stats.get('total_entries', 0)} 条记录")

        logger.info(f"翻译完成: {translated_path}")
        return translated_path


class VideoEmbedder:
    """视频字幕嵌入器"""
    
    def __init__(self, config: Config):
        self.config = config

    def _parse_time(self, time_str: str) -> float:
        """将 FFmpeg 进度时间字符串转换为秒数"""
        return parse_time(time_str)

    def embed(self, video_path: str, srt_path: str, output_path: str):
        """将字幕硬嵌入视频（带进度条）"""
        video_size_gb = get_video_size_gb(video_path)
        duration = get_video_duration(video_path)
        start_time = time.time()

        logger.info(f"开始嵌入字幕到视频")
        logger.info(f"输入视频: {video_path} ({video_size_gb:.2f} GB, {duration:.1f}秒)")
        logger.info(f"字幕文件: {srt_path}")
        logger.info(f"输出视频: {output_path}")

        # 转义字幕路径（FFmpeg 滤镜语法要求）
        srt_path_escaped = srt_path.replace('\\', '/').replace(':', '\\:')

        subtitle_filter = (
            f"subtitles='{srt_path_escaped}':"
            f"force_style='{self.config.SUBTITLE_STYLE},MarginV={self.config.SUBTITLE_MARGIN_V}'"
        )

        cmd = [
            'ffmpeg', '-y', '-progress', 'pipe:1', '-i', video_path,
            '-vf', subtitle_filter,
            '-c:v', self.config.VIDEO_CODEC,
        ]

        # 根据编码器设置参数
        if self.config.VIDEO_CODEC == 'h264_nvenc':
            cmd.extend(['-preset', 'medium', '-cq', str(self.config.VIDEO_CRF)])
        elif self.config.VIDEO_CODEC == 'libx264':
            cmd.extend(['-preset', 'medium', '-crf', str(self.config.VIDEO_CRF)])
        else:
            logger.warning(f"未识别的视频编码器: {self.config.VIDEO_CODEC}，将使用默认参数")
            cmd.extend(['-preset', 'medium', '-crf', str(self.config.VIDEO_CRF)])

        cmd.extend([
            '-c:a', self.config.AUDIO_CODEC,
            '-b:a', self.config.AUDIO_BITRATE,
        ])

        cmd.append(output_path)

        logger.info("正在编码视频，这可能需要一些时间...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1,
        )

        current_time = 0.0
        last_progress_log = 0
        stderr_lines = []

        def read_stderr():
            while process.poll() is None:
                line = process.stderr.readline()
                if line:
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue

                line = line.strip()

                if line.startswith('out_time_ms='):
                    try:
                        current_time = int(line.split('=')[1]) / 1000000.0
                    except (ValueError, IndexError):
                        continue
                elif line.startswith('out_time='):
                    try:
                        time_str = line.split('=')[1]
                        if time_str and time_str != 'N/A':
                            current_time = self._parse_time(time_str)
                    except (ValueError, IndexError):
                        continue

                current_timestamp = time.time()
                if duration > 0 and (current_timestamp - last_progress_log) >= 5:
                    progress = min(100.0, (current_time / duration) * 100)
                    logger.info(f"编码进度: {progress:.1f}% ({current_time:.1f}s / {duration:.1f}s)")
                    last_progress_log = current_timestamp

            process.wait()
            stderr_thread.join(timeout=2)

            if process.returncode != 0:
                stderr_output = ''.join(stderr_lines[-20:])
                if stderr_output.strip():
                    logger.error(f"FFmpeg 错误输出:\n{stderr_output.strip()}")
                raise RuntimeError(f"视频编码失败 (exit code: {process.returncode})")

            elapsed = time.time() - start_time
            logger.info(f"视频嵌入完成: {output_path} (耗时: {elapsed:.1f}秒)")

        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
            raise RuntimeError("用户中断视频编码")


# ==================== 主流程 ====================

class VideoSubtitlePipeline:
    """视频字幕处理流程"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.extractor = AudioExtractor(self.config)
        self.generator = SubtitleGenerator(self.config)
        self.translator = SubtitleTranslator(self.config)
        self.embedder = VideoEmbedder(self.config)
    
    def process(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        skip_transcription: bool = False,
        skip_translation: bool = False,
        skip_embedding: bool = False
    ) -> dict:
        """
        处理视频完整流程
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            skip_transcription: 跳过语音识别
            skip_translation: 跳过翻译
            skip_embedding: 跳过嵌入字幕
        """
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        if output_path is None:
            video_dir = os.path.dirname(video_path)
            video_name = Path(video_path).stem
            output_path = os.path.join(video_dir, f"{video_name}_subtitled.mp4")
        
        result = {
            'video_path': video_path,
            'output_path': output_path,
            'audio_segments': None,
            'original_srt': None,
            'translated_srt': None,
            'success': False
        }
        
        try:
            # 步骤 1: 提取音频（支持分段）
            if not skip_transcription:
                audio_segments = self.extractor.extract(video_path)
                result['audio_segments'] = audio_segments
            else:
                video_name = Path(video_path).stem
                existing_srt = os.path.join(
                    self.config.TEMP_DIR, 
                    f"{video_name}_original.srt"
                )
                if os.path.exists(existing_srt):
                    result['original_srt'] = existing_srt
                    logger.info(f"使用已存在的字幕: {existing_srt}")
                else:
                    raise FileNotFoundError(f"找不到已存在的字幕文件: {existing_srt}")
            
            # 步骤 2: 生成字幕（支持多段音频）
            if not skip_transcription:
                srt_path = self.generator.generate(audio_segments, video_path)
                result['original_srt'] = srt_path
            else:
                srt_path = result['original_srt']
            
            # 检查字幕是否为空
            if srt_path and os.path.exists(srt_path):
                with open(srt_path, 'r', encoding='utf-8-sig') as f:
                    srt_content = f.read().strip()
                if not srt_content:
                    logger.warning("未识别到任何字幕，直接复制原视频")
                    if not skip_embedding:
                        # 复制原视频到输出路径
                        shutil.copy2(video_path, output_path)
                        logger.info(f"原视频已复制到: {output_path}")
                    result['success'] = True
                    return result
            
            # 步骤 3: 翻译字幕（批量切片）
            if not skip_translation:
                translated_srt = self.translator.translate_srt(srt_path, video_path)
                result['translated_srt'] = translated_srt
            else:
                result['translated_srt'] = srt_path
            
            # 步骤 4: 嵌入字幕（支持大视频分段）
            if not skip_embedding:
                self.embedder.embed(video_path, result['translated_srt'], output_path)
            
            result['success'] = True
            logger.info("=" * 50)
            logger.info("处理完成！")
            logger.info(f"输出文件: {output_path}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise
        
        finally:
            # 只清理音频临时文件，保留字幕文件供用户查看
            if result.get('audio_segments') and not self.config.KEEP_TEMP:
                for audio_path in result['audio_segments']:
                    if os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                            logger.info(f"已清理临时文件: {audio_path}")
                        except Exception as e:
                            logger.warning(f"清理临时文件失败: {audio_path} - {e}")
        
        return result


# ==================== 命令行接口 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='视频自动字幕生成与翻译工具（支持长视频分段处理）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（默认百炼平台）
  python video_subtitle.py input.mp4

  # 使用 DeepSeek 翻译
  python video_subtitle.py input.mp4 --provider deepseek

  # 测试 API 连通性
  python video_subtitle.py --test-api --provider siliconflow

  # 指定音频分段长度
  python video_subtitle.py input.mp4 --audio-segment 20

  # 调整翻译批量大小
  python video_subtitle.py input.mp4 --batch-size 50

  # 跳过翻译（使用原始语言字幕）
  python video_subtitle.py input.mp4 --skip-translation

  # CPU 模式
  python video_subtitle.py input.mp4 --device cpu

  # 翻译为英文
  python video_subtitle.py input.mp4 --target-language English
        """
    )
    
    parser.add_argument('video', nargs='?', help='输入视频文件路径（--test-api 模式下可选）')
    parser.add_argument('-o', '--output', help='输出视频文件路径')
    parser.add_argument('--skip-transcription', action='store_true', 
                        help='跳过语音识别')
    parser.add_argument('--skip-translation', action='store_true',
                        help='跳过翻译步骤')
    parser.add_argument('--skip-embedding', action='store_true',
                        help='跳过嵌入字幕')
    parser.add_argument('--model', default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'turbo', 'large-v1', 'large-v2', 'large-v3'],
                        help='Whisper 模型大小（默认: large-v3）')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备（默认: cuda）')
    parser.add_argument('--language', help='源语言代码（默认自动检测）')
    parser.add_argument('--target-language', default='中文',
                        help='目标翻译语言（默认: 中文）')
    parser.add_argument('--audio-segment', type=int, default=30,
                        help='音频分段长度（分钟，默认: 30）')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='翻译批量大小（行数，默认: 30）')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用翻译缓存')
    parser.add_argument('--test-api', action='store_true',
                        help='测试 API 连通性（不处理视频）')
    parser.add_argument('--provider', default='bailian',
                        choices=['bailian', 'siliconflow', 'deepseek'],
                        help='翻译平台（默认: bailian）')
    
    args = parser.parse_args()
    
    # 创建配置（加载 .env 文件中的环境变量）
    config = Config()
    
    # 测试 API 连通性
    if args.test_api:
        translator = SubtitleTranslator(config)
        success = translator.test_connection()
        sys.exit(0 if success else 1)
    
    # 非测试模式下需要视频文件
    if not args.video:
        parser.error("需要指定视频文件路径，或使用 --test-api 测试 API 连通性")
    
    # 检查 ffmpeg
    if not check_ffmpeg():
        logger.error("未检测到 ffmpeg，请先安装: https://ffmpeg.org/download.html")
        sys.exit(1)
    config.WHISPER_MODEL = args.model
    config.DEVICE = args.device
    config.AUDIO_SEGMENT_MINUTES = args.audio_segment
    config.TRANSLATION_BATCH_SIZE = args.batch_size
    config.TRANSLATION_PROVIDER = args.provider
    config.TARGET_LANGUAGE = args.target_language
    config.ENABLE_CACHE = not args.no_cache
    if args.language:
        config.SOURCE_LANGUAGE = args.language
    
    # 检查设备配置
    if config.DEVICE == 'cpu':
        # CPU 模式自动使用 int8（CPU 不支持 float16）
        if config.COMPUTE_TYPE == 'float16':
            logger.info("CPU 模式自动切换计算类型为 int8")
            config.COMPUTE_TYPE = 'int8'
        # CPU 模式下使用软件编码
        if config.VIDEO_CODEC == 'h264_nvenc':
            logger.info("CPU 模式自动切换视频编码器为 libx264")
            config.VIDEO_CODEC = 'libx264'
    elif config.DEVICE == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，自动切换到 CPU 模式")
                config.DEVICE = 'cpu'
                config.COMPUTE_TYPE = 'int8'
                config.VIDEO_CODEC = 'libx264'
        except ImportError:
            pass
    
    # 执行处理
    pipeline = VideoSubtitlePipeline(config)
    
    try:
        result = pipeline.process(
            args.video,
            args.output,
            skip_transcription=args.skip_transcription,
            skip_translation=args.skip_translation,
            skip_embedding=args.skip_embedding
        )
        
        if result['success']:
            logger.info("=" * 50)
            logger.info("处理成功！")
            logger.info(f"原始字幕: {result.get('original_srt', 'N/A')}")
            logger.info(f"翻译字幕: {result.get('translated_srt', 'N/A')}")
            if not args.skip_embedding:
                logger.info(f"输出视频: {result['output_path']}")
            logger.info("=" * 50)
            
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
