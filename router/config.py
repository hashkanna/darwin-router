import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Hackathon workspace MaaS host (from onboarding doc), not the public intl URL
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://ws-217y1bpliyzcf5nl.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")

SIE_BASE_URL = os.getenv("SIE_BASE_URL", "https://api.superlinked.com/v1").rstrip("/")
SIE_API_KEY = os.getenv("SIE_API_KEY", "")
SIE_EMBED_MODEL = os.getenv("SIE_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
SIE_CHAT_MODEL = os.getenv("SIE_CHAT_MODEL", "Qwen/Qwen3.5-4B")
SIE_SMART_MODEL = os.getenv("SIE_SMART_MODEL", "Qwen/Qwen3.6-27B")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
EXEMPLARS_PATH = DATA_DIR / "exemplars.jsonl"
LOG_PATH = DATA_DIR / "log.jsonl"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
JUDGE_LOG_PATH = DATA_DIR / "judge.jsonl"
REPLAY_LOG_PATH = DATA_DIR / "replay.jsonl"
HOLDOUT_PATH = DATA_DIR / "holdout.jsonl"

TEXT_ROUTES = ["simple", "reasoning"]

# simple runs on SIE (cheap lane); reasoning/vision offload to Alibaba Cloud —
# the judged "offload" pattern. Degrade mode: whichever key is missing, routes
# fall back to the other backend (SIE Qwen3.6-27B covers reasoning/vision, it
# takes text+image), so the router works end-to-end with either key alone.
MODEL_FOR_ROUTE = {
    "simple": os.getenv("SIMPLE_MODEL", SIE_CHAT_MODEL if SIE_API_KEY else "qwen-flash"),
    "reasoning": os.getenv("REASONING_MODEL", "qwen-max" if DASHSCOPE_API_KEY else SIE_SMART_MODEL),
    "vision": os.getenv("VISION_MODEL", "qwen-vl-max" if DASHSCOPE_API_KEY else SIE_SMART_MODEL),
    # z-image-turbo: ~8s sync, demo-friendly; qwen-image-3.0-pro is slower/nicer
    "imagegen": os.getenv("IMAGEGEN_MODEL", "z-image-turbo"),
    "videogen": "happyhorse-1.1-t2v",
}

# Visual models use the DashScope NATIVE api on dashscope-intl (Singapore),
# not the workspace compatible-mode host. Image gen is sync; video is async
# (X-DashScope-Async: enable, then poll /tasks/{task_id}).
DASHSCOPE_NATIVE_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
IMAGEGEN_ENDPOINT = f"{DASHSCOPE_NATIVE_BASE}/services/aigc/multimodal-generation/generation"
VIDEOGEN_ENDPOINT = f"{DASHSCOPE_NATIVE_BASE}/services/aigc/video-generation/video-synthesis"
BACKEND_FOR_ROUTE = {
    "simple": "sie" if SIE_API_KEY else "modelstudio",
    "reasoning": "modelstudio" if DASHSCOPE_API_KEY else "sie",
    "vision": "modelstudio" if DASHSCOPE_API_KEY else "sie",
    "imagegen": "modelstudio",
    "videogen": "modelstudio",
}
