"""下载本地后端 (--backend local/worker) 所需的模型、二进制和第三方源码。

默认全部下载（模型优先走 ModelScope，失败或不存在时回退 HuggingFace）：
  models/Fun-CosyVoice3-0.5B      ModelScope FunAudioLLM/Fun-CosyVoice3-0.5B-2512
                                  (回退 HF FunAudioLLM/Fun-CosyVoice3-0.5B)
  models/Qwen3.5-2B-Q8_0.gguf     ModelScope unsloth/Qwen3.5-2B-GGUF (回退 HF 同名仓库)
  bin/llama/                      llama.cpp GitHub release (win-cpu-x64)
  third_party/CosyVoice-main/     GitHub FunAudioLLM/CosyVoice (+ Matcha-TTS)
  ~/.cache/modelscope/...         ModelScope 流式 Paraformer（ASR）

用法：
  uv run --extra gpu python scripts/fetch_deps.py                 # 下载全部
  uv run --extra gpu python scripts/fetch_deps.py --only llama
  uv run --extra gpu python scripts/fetch_deps.py --check         # 只检查，不下载
  uv run --extra gpu python scripts/fetch_deps.py --dry-run       # 只显示计划
  uv run --extra gpu python scripts/fetch_deps.py --source hf     # 模型全部走 HuggingFace

国内网络可设置镜像：NV_HF_ENDPOINT=https://hf-mirror.com（--source hf 时生效）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
HF_ENDPOINT = os.environ.get("NV_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MS_ENDPOINT = os.environ.get("NV_MS_ENDPOINT", "https://www.modelscope.cn").rstrip("/")
MS_REVISION = "master"

TTS_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B"
TTS_MS_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TTS_DEST = ROOT / "models" / "Fun-CosyVoice3-0.5B"
# 只取推理需要的文件，跳过 flow.decoder.estimator.fp32.onnx / llm.rl.pt 等约 3.3GB。
TTS_FILES = [
    "configuration.json",
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "flow.pt",
    "hift.pt",
    "llm.pt",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
]

LLM_REPO = "unsloth/Qwen3.5-2B-GGUF"
LLM_MS_REPO = "unsloth/Qwen3.5-2B-GGUF"
LLM_FILE = "Qwen3.5-2B-Q8_0.gguf"
LLM_DEST = ROOT / "models" / LLM_FILE

COSYVOICE_REPO = "FunAudioLLM/CosyVoice"
COSYVOICE_REF = "main"
MATCHA_REPO = "shivammehta25/Matcha-TTS"
MATCHA_REF = "main"
THIRD_PARTY_DEST = ROOT / "third_party" / "CosyVoice-main"

LLAMA_BUILD = "b12125"  # 与仓库验证过的 llama-server 版本一致 (0.9.0-dev build 12125)
LLAMA_ASSET = "llama-{build}-bin-win-cpu-x64.zip"
LLAMA_DEST = ROOT / "bin" / "llama"

ASR_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models"

UA = "native-voice-fetch-deps/1.0"

_MS_CACHE: dict[str, bool] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}GB"


def open_url(url: str, headers: dict[str, str] | None = None, retries: int = 3):
    req_headers = {"User-Agent": UA}
    if headers:
        req_headers.update(headers)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            return urllib.request.urlopen(req, timeout=60)  # noqa: S310
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"请求失败: {url} ({last})")


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            value = resp.headers.get("Content-Length")
            return int(value) if value and value.isdigit() else None
    except Exception:
        return None


def download(url: str, dest: Path, expected: int | None = None, force: bool = False) -> None:
    """带断点续传的下载；已存在且大小正确则跳过。"""
    if dest.exists() and not force:
        if expected is None or dest.stat().st_size == expected:
            log(f"  已存在 {dest.relative_to(ROOT)}")
            return
        log(f"  大小不符，重新下载 {dest.relative_to(ROOT)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if force and part.exists():
        part.unlink()
    if expected is None:
        expected = remote_size(url)
    start = part.stat().st_size if part.exists() else 0
    if expected is not None and start >= expected:
        part.replace(dest)
        log(f"  已存在 {dest.relative_to(ROOT)}")
        return
    headers = {"Range": f"bytes={start}-"} if start else {}
    with open_url(url, headers) as resp:
        if start and resp.status != 206:  # 服务器不支持续传
            start = 0
        total = expected or (int(resp.headers.get("Content-Length", 0)) + start or None)
        mode = "ab" if start else "wb"
        done = start
        last_log = 0.0
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_log > 1.0:
                    last_log = now
                    pct = f" {done * 100 // total:3d}%" if total else ""
                    log(f"  {dest.name}: {human(done)}{pct}")
    if expected is not None and part.stat().st_size != expected:
        raise RuntimeError(
            f"{dest.name} 大小不符：得到 {part.stat().st_size}，期望 {expected}（可重试或加 --force）"
        )
    part.replace(dest)
    log(f"  完成 {dest.relative_to(ROOT)} ({human(dest.stat().st_size)})")


def extract_zip(archive: Path, dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n and not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        for name in names:
            rel = name.split("/", 1)[1] if strip else name
            target = dest / rel
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)


def fetch_zip(url: str, dest: Path, strip_hint: str, force: bool, dry_run: bool) -> None:
    archive = ROOT / ".data" / "downloads" / url.rsplit("/", 1)[-1]
    size = remote_size(url)
    log(f"  下载 {url}" + (f" ({human(size)})" if size else ""))
    if dry_run:
        return
    download(url, archive, size, force)
    extract_zip(archive, dest, force)
    archive.unlink(missing_ok=True)
    if not (dest / strip_hint).exists():
        raise RuntimeError(f"解压后缺少 {dest / strip_hint}")


def step_third_party(args) -> None:
    log("[third-party] CosyVoice 源码")
    if not args.force and (THIRD_PARTY_DEST / "cosyvoice" / "cli" / "cosyvoice.py").exists():
        log(f"  已存在 {THIRD_PARTY_DEST.relative_to(ROOT)}")
        return
    url = f"https://github.com/{COSYVOICE_REPO}/archive/refs/heads/{COSYVOICE_REF}.zip"
    fetch_zip(url, THIRD_PARTY_DEST, "cosyvoice/cli/cosyvoice.py", args.force, args.dry_run)
    matcha_dest = THIRD_PARTY_DEST / "third_party" / "Matcha-TTS"
    url = f"https://github.com/{MATCHA_REPO}/archive/refs/heads/{MATCHA_REF}.zip"
    fetch_zip(url, matcha_dest, "matcha/models/components/flow_matching.py", args.force, args.dry_run)


def ms_repo_exists(repo_id: str) -> bool:
    if repo_id in _MS_CACHE:
        return _MS_CACHE[repo_id]
    try:
        with open_url(f"{MS_ENDPOINT}/api/v1/models/{repo_id}", retries=2) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    _MS_CACHE[repo_id] = ok
    return ok


def resolve_source(args, hf_repo: str, ms_repo: str | None) -> tuple[str, str]:
    """返回 (source, repo_id)；auto 时优先 ModelScope，其次 HuggingFace。"""
    if args.source == "hf":
        return "hf", hf_repo
    if args.source == "modelscope":
        if not ms_repo:
            raise SystemExit("该模型没有 ModelScope 仓库，请用 --source hf")
        return "modelscope", ms_repo
    if ms_repo and ms_repo_exists(ms_repo):
        return "modelscope", ms_repo
    log(f"  ModelScope 上找不到 {ms_repo or hf_repo}，回退 HuggingFace")
    return "hf", hf_repo


def file_url(source: str, repo_id: str, path: str) -> str:
    if source == "modelscope":
        return (
            f"{MS_ENDPOINT}/api/v1/models/{repo_id}/repo"
            f"?Revision={MS_REVISION}&FilePath={quote(path)}"
        )
    return f"{HF_ENDPOINT}/{repo_id}/resolve/main/{quote(path)}"


def step_model(name: str, args, hf_repo: str, ms_repo: str | None, dest: Path, files: list[str]) -> None:
    source, repo = resolve_source(args, hf_repo, ms_repo)
    log(f"[{name}] {source}:{repo} -> {dest.relative_to(ROOT)}")
    for rel in files:
        url = file_url(source, repo, rel)
        size = remote_size(url)
        log(f"  {rel}" + (f" ({human(size)})" if size else ""))
        if args.dry_run:
            continue
        try:
            download(url, dest / rel, size, args.force)
        except Exception as exc:
            if source != "modelscope" or args.source == "modelscope":
                raise
            log(f"  ModelScope 下载失败（{exc}），改用 HuggingFace")
            alt = file_url("hf", hf_repo, rel)
            download(alt, dest / rel, remote_size(alt), args.force)


def step_tts(args) -> None:
    step_model("tts", args, TTS_REPO, TTS_MS_REPO, TTS_DEST, TTS_FILES)


def step_llm(args) -> None:
    step_model("llm", args, LLM_REPO, LLM_MS_REPO, LLM_DEST.parent, [LLM_FILE])


def step_llama(args) -> None:
    log(f"[llama] llama.cpp {args.llama_build} -> {LLAMA_DEST.relative_to(ROOT)}")
    if not args.force and (LLAMA_DEST / "llama-server.exe").exists() and not args.dry_run:
        log(f"  已存在 {LLAMA_DEST.relative_to(ROOT)}")
        return
    asset = LLAMA_ASSET.format(build=args.llama_build)
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{args.llama_build}/{asset}"
    fetch_zip(url, LLAMA_DEST, "llama-server.exe", args.force, args.dry_run)


def step_asr(args) -> None:
    log(f"[asr] {ASR_MODEL}")
    cached = MODELSCOPE_CACHE / ASR_MODEL
    if not args.force and (cached / "model.pt").exists():
        log(f"  已存在 {cached}")
        return
    if args.dry_run:
        log("  通过 modelscope 下载到 " + str(cached))
        return
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise SystemExit("缺少 modelscope，请先运行: uv sync --extra gpu") from None
    path = snapshot_download(ASR_MODEL)
    log(f"  完成 {path}")


STEPS = {
    "third-party": step_third_party,
    "tts": step_tts,
    "llm": step_llm,
    "llama": step_llama,
    "asr": step_asr,
}

# --check 用的关键文件
REQUIRED = {
    "third-party": [
        THIRD_PARTY_DEST / "cosyvoice" / "cli" / "cosyvoice.py",
        THIRD_PARTY_DEST / "third_party/Matcha-TTS/matcha/models/components/flow_matching.py",
        THIRD_PARTY_DEST / "asset" / "zero_shot_prompt.wav",
    ],
    "tts": [TTS_DEST / rel for rel in TTS_FILES],
    "llm": [LLM_DEST],
    "llama": [LLAMA_DEST / "llama-server.exe"],
    "asr": [MODELSCOPE_CACHE / ASR_MODEL / "model.pt"],
}


def check(names: list[str] | None = None) -> int:
    missing = 0
    for name, paths in REQUIRED.items():
        if names is not None and name not in names:
            continue
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                log(f"缺少 {path}")
                missing += 1
    if missing:
        log(f"\n共 {missing} 个文件缺失，运行 uv run --extra gpu python scripts/fetch_deps.py 下载")
        return 1
    log("本地依赖齐全")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 native-voice 本地模型/二进制/第三方源码")
    parser.add_argument("--only", default=",".join(STEPS), help=f"逗号分隔，可选: {','.join(STEPS)}")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--dry-run", action="store_true", help="只打印下载计划")
    parser.add_argument("--check", action="store_true", help="只检查本地依赖是否齐全")
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "hf"],
        default=os.environ.get("NV_MODEL_SOURCE", "auto"),
        help="模型下载源：auto=ModelScope 优先、HF 回退（默认）",
    )
    parser.add_argument("--llama-build", default=LLAMA_BUILD, help=f"llama.cpp 构建号 (默认 {LLAMA_BUILD})")
    args = parser.parse_args()

    if args.check:
        return check()

    selected = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        parser.error(f"未知的组件: {','.join(unknown)}（可选: {','.join(STEPS)}）")

    log(f"模型源: {args.source}  (ModelScope={MS_ENDPOINT}, HuggingFace={HF_ENDPOINT})"
        + ("  (dry-run)" if args.dry_run else ""))
    started = time.monotonic()
    for name in selected:
        STEPS[name](args)
    if not args.dry_run:
        log(f"\n完成，用时 {time.monotonic() - started:.0f}s")
        return check(selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
