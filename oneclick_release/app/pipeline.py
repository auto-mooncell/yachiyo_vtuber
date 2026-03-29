import argparse
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import wave
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent
EASY_ROOT = WORKSPACE_ROOT / "EasyVtuber"
CONFIG_PATH = RELEASE_ROOT / "config" / "defaults.json"
SESSION_PATH = RELEASE_ROOT / "data" / "output" / "current_session.json"
LOG_DIR = RELEASE_ROOT / "data" / "output" / "logs"
PID_PATH = RELEASE_ROOT / "data" / "output" / "pids.json"
LIST_DIR = RELEASE_ROOT / "data" / "working" / "lists"
CLEAN_AUDIO_DIR = RELEASE_ROOT / "data" / "working" / "clean_audio"
IMAGE_INPUT_DIR = RELEASE_ROOT / "data" / "input" / "image"
AUDIO_INPUT_DIR = RELEASE_ROOT / "data" / "input" / "audio"
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 10.0
SUPPORTED_LANGUAGES = {"ja", "zh", "en", "yue", "auto"}

CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
LOCAL_TEXT_ENCODING = locale.getpreferredencoding(False) or "utf-8"


def resolve_gpt_root():
    candidates = [
        WORKSPACE_ROOT / "GPT-SoVITS",
        WORKSPACE_ROOT / "GPT-SoVITS-v2pro-20250604",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


GPT_ROOT = resolve_gpt_root()


def known_gpt_roots():
    roots = [GPT_ROOT, WORKSPACE_ROOT / "GPT-SoVITS", WORKSPACE_ROOT / "GPT-SoVITS-v2pro-20250604"]
    unique = []
    seen = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def clone_config(cfg):
    return json.loads(json.dumps(cfg))


def gpt_mode(cfg):
    return cfg["gpt_sovits"].get("mode", "local").lower()


def using_remote_gpt(cfg):
    return gpt_mode(cfg) == "remote"


def print_command(command):
    printable = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)
    print(f"\n$ {printable}\n")


def run_checked(command, cwd, env=None):
    print_command(command)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def start_detached(command, cwd, env=None, stdout_path=None, stderr_path=None):
    print_command(command)
    stdout_handle = None
    stderr_handle = None
    if stdout_path:
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = open(stdout_path, "a", encoding="utf-8", errors="replace")
    if stderr_path:
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = open(stderr_path, "a", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        creationflags=CREATE_NEW_CONSOLE,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )


def load_pids():
    if not PID_PATH.exists():
        return {}
    return json.loads(PID_PATH.read_text(encoding="utf-8"))


def remap_to_current_gpt_root(path_value):
    if not path_value:
        return path_value
    raw_path = Path(path_value)
    candidate = raw_path
    if not raw_path.is_absolute():
        candidate = (GPT_ROOT / raw_path).resolve()
        return str(candidate)
    for root in known_gpt_roots():
        try:
            relative = raw_path.relative_to(root)
        except ValueError:
            continue
        return str((GPT_ROOT / relative).resolve())
    return str(raw_path)


def normalize_session_paths(session):
    normalized = dict(session)
    for key in ("list_path", "gpt_weights", "sovits_weights", "reference_audio"):
        if key in normalized:
            normalized[key] = remap_to_current_gpt_root(normalized[key])
    return normalized


def save_pids(data):
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record_pid(name, process):
    pids = load_pids()
    pids[name] = process.pid
    save_pids(pids)


def clear_pid(name):
    pids = load_pids()
    if name in pids:
        del pids[name]
        save_pids(pids)


def process_alive(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}"],
            check=False,
            capture_output=True,
            text=True,
            encoding=LOCAL_TEXT_ENCODING,
            errors="replace",
        )
        return str(int(pid)) in result.stdout
    except Exception:
        return False


def find_listener_pid(port):
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            check=False,
            capture_output=True,
            text=True,
            encoding=LOCAL_TEXT_ENCODING,
            errors="replace",
        )
        needle = f":{int(port)}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(needle) and parts[3] == "LISTENING":
                return int(parts[4])
    except Exception:
        return None
    return None


def stop_pid(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_easyvtuber_debug_frame_pids():
    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process -ErrorAction SilentlyContinue | "
            "Where-Object { $_.MainWindowTitle -eq 'EasyVtuber Debug Frame' } | "
            "Select-Object -ExpandProperty Id",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding=LOCAL_TEXT_ENCODING,
            errors="replace",
        )
        stdout = result.stdout
        pids = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids
    except Exception:
        return []


def stop_tracked_processes():
    cfg = load_config()
    pids = load_pids()
    for name, pid in list(pids.items()):
        if process_alive(pid):
            print(f"Stopping {name} (PID {pid})")
            stop_pid(pid)
        clear_pid(name)
    for pid in find_easyvtuber_debug_frame_pids():
        if process_alive(pid):
            print(f"Stopping orphan EasyVtuber Debug Frame (PID {pid})")
            stop_pid(pid)
    if not using_remote_gpt(cfg):
        api_pid = find_listener_pid(cfg["gpt_sovits"]["api_port"])
        if api_pid and process_alive(api_pid):
            print(f"Stopping GPT-SoVITS API listener (PID {api_pid})")
            stop_pid(api_pid)


def print_status():
    pids = load_pids()
    if not pids:
        print("No tracked background processes.")
        return
    for name, pid in pids.items():
        print(f"{name}: PID {pid}, alive={process_alive(pid)}")


def ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_language(value, fallback=None):
    if value is None:
        return fallback
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "jp": "ja",
        "jpn": "ja",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-tw": "zh",
        "zh-hant": "zh",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    return fallback


def config_with_language(cfg, language):
    updated = clone_config(cfg)
    if language:
        updated["gpt_sovits"]["language"] = language
        updated["bridge"]["text_lang"] = language
    return updated


def find_single_file(directory, extensions):
    ensure_exists(directory, "Input directory")
    matches = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            matches.append(item)
    if not matches:
        raise FileNotFoundError(f"No matching input files in {directory}")
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    if len(matches) > 1:
        print(f"Multiple files found in {directory}; selecting newest: {matches[0].name}")
    return matches[0]


def find_character_input_file(directory, character_name, extensions, label):
    ensure_exists(directory, "Input directory")
    matches = [item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in extensions]
    if not matches:
        raise FileNotFoundError(f"No matching {label} files in {directory}")
    exact_match = directory / f"{character_name}{next(iter(sorted(extensions)))}"
    normalized_character = character_name.casefold()
    same_stem_matches = [item for item in matches if item.stem.casefold() == normalized_character]
    if same_stem_matches:
        same_stem_matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return same_stem_matches[0]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(sorted(item.name for item in matches))
    raise FileNotFoundError(
        f"Multiple {label} files found in {directory}, but none match character '{character_name}'. "
        f"Rename the desired file to '{character_name}<ext>' or keep only one file. Available: {available}"
    )


def should_remove_bg(args):
    return bool(getattr(args, "remove_bg", False) or getattr(args, "remove", None) == "bg")


def choose_character_name(args):
    if args.character:
        return args.character
    image_path = find_single_file(IMAGE_INPUT_DIR, {".png"})
    return image_path.stem


def copy_character_image(character_name):
    image_path = find_character_input_file(IMAGE_INPUT_DIR, character_name, {".png"}, "image")
    target_path = EASY_ROOT / "data" / "images" / f"{character_name}.png"
    shutil.copyfile(image_path, target_path)
    print(f"Character image installed: {target_path}")
    return target_path


def resolve_python_executable(preferred_path, windowless=False):
    if preferred_path.exists():
        return preferred_path
    current_python = Path(sys.executable)
    if windowless:
        pythonw_candidate = current_python.with_name("pythonw.exe")
        if pythonw_candidate.exists():
            return pythonw_candidate
    return current_python


def build_runtime_python(windowless=False):
    name = "pythonw.exe" if windowless else "python.exe"
    return resolve_python_executable(GPT_ROOT / "runtime" / name, windowless=windowless)


def build_easy_python(windowless=False):
    name = "pythonw.exe" if windowless else "python.exe"
    return resolve_python_executable(EASY_ROOT / ".venv" / "Scripts" / name, windowless=windowless)


def remove_background_audio(character_name, audio_path):
    runtime_python = build_runtime_python()
    ensure_exists(runtime_python, "GPT-SoVITS runtime python")
    CLEAN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = CLEAN_AUDIO_DIR / f"{character_name}_clean.wav"
    run_checked(
        [
            str(runtime_python),
            "tools/remove_bg.py",
            "--input",
            str(audio_path),
            "--output",
            str(cleaned_path),
        ],
        GPT_ROOT,
    )
    ensure_exists(cleaned_path, "Background-removed audio")
    return cleaned_path


def resolve_training_audio(character_name, args):
    audio_path = find_character_input_file(AUDIO_INPUT_DIR, character_name, {".wav", ".mp3", ".flac", ".m4a", ".ogg"}, "audio")
    if should_remove_bg(args):
        print(f"Removing background audio before training: {audio_path.name}")
        return remove_background_audio(character_name, audio_path)
    return audio_path


def find_generated_asr_list(asr_dir, character_name, slicer_dir):
    candidates = [
        asr_dir / f"{character_name}.list",
        asr_dir / f"{slicer_dir.name}.list",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(asr_dir.glob("*.list"))
    if matches:
        matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        newest = matches[0]
        if newest.stat().st_mtime >= slicer_dir.stat().st_mtime:
            return newest
    return candidates[0]


def slice_and_asr(character_name, cfg, audio_path=None):
    if audio_path is None:
        audio_path = find_character_input_file(
            AUDIO_INPUT_DIR, character_name, {".wav", ".mp3", ".flac", ".m4a", ".ogg"}, "audio"
        )
    slicer_dir = GPT_ROOT / "output" / "slicer_opt" / character_name
    asr_dir = GPT_ROOT / "output" / "asr_opt"
    runtime_python = build_runtime_python()
    ensure_exists(runtime_python, "GPT-SoVITS runtime python")

    if slicer_dir.exists():
        shutil.rmtree(slicer_dir)
    stale_asr_list = asr_dir / f"{character_name}.list"
    if stale_asr_list.exists():
        stale_asr_list.unlink()

    run_checked(
        [
            str(runtime_python),
            "tools/slice_audio.py",
            str(audio_path),
            str(slicer_dir),
            "-40",
            "5000",
            "300",
            "10",
            "500",
            "0.9",
            "0.5",
            "0",
            "1",
        ],
        GPT_ROOT,
    )

    language = cfg["gpt_sovits"]["language"]
    asr_precision = cfg["gpt_sovits"]["asr_precision"]
    if language in {"zh", "yue"}:
        command = [
            str(runtime_python),
            "tools/asr/funasr_asr.py",
            "-i",
            str(slicer_dir),
            "-o",
            str(asr_dir),
            "-l",
            language,
            "-p",
            asr_precision,
        ]
    else:
        command = [
            str(runtime_python),
            "tools/asr/fasterwhisper_asr.py",
            "-i",
            str(slicer_dir),
            "-o",
            str(asr_dir),
            "-l",
            language,
            "-p",
            asr_precision,
            "-s",
            cfg["gpt_sovits"]["asr_model_size"],
        ]
    try:
        run_checked(command, GPT_ROOT)
    except subprocess.CalledProcessError as exc:
        source_list = find_generated_asr_list(asr_dir, character_name, slicer_dir)
        if source_list.exists():
            print(
                "ASR process returned a non-zero exit code, but the list file was already generated. "
                f"Continuing with: {source_list}"
            )
        else:
            raise exc

    source_list = find_generated_asr_list(asr_dir, character_name, slicer_dir)
    ensure_exists(source_list, "Generated list")
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    target_list = LIST_DIR / f"{character_name}.list"
    shutil.copyfile(source_list, target_list)
    print(f"Review list ready: {target_list}")
    return target_list


def review_list(list_path):
    subprocess.Popen(["notepad.exe", str(list_path)])
    input("Review and correct the list, then press Enter to continue...")


def train_character(character_name, list_path, cfg):
    runtime_python = build_runtime_python()
    training_cfg = cfg.get("training", {})
    run_checked(
        [
            str(runtime_python),
            "train_character.py",
            "--exp-name",
            character_name,
            "--list-file",
            str(list_path),
            "--version",
            cfg["gpt_sovits"]["version"],
            "--language",
            cfg["gpt_sovits"]["language"],
            "--gpu",
            cfg["gpt_sovits"]["gpu"],
            "--s2-batch-size",
            str(training_cfg.get("s2_batch_size", 4)),
            "--s2-epochs",
            str(training_cfg.get("s2_epochs", 8)),
            "--s2-save-every-epoch",
            str(training_cfg.get("s2_save_every_epoch", 4)),
            "--s1-batch-size",
            str(training_cfg.get("s1_batch_size", 1)),
            "--s1-epochs",
            str(training_cfg.get("s1_epochs", 15)),
            "--s1-save-every-n-epoch",
            str(training_cfg.get("s1_save_every_n_epoch", 5)),
        ],
        GPT_ROOT,
    )    


def find_character_list(character_name):
    candidates = [
        LIST_DIR / f"{character_name}.list",
        GPT_ROOT / "output" / "asr_opt" / f"{character_name}.list",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No list file found for character: {character_name}")


def find_latest_weight(directory, pattern):
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No weights found for pattern: {pattern}")
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def get_reference_duration_seconds(audio_path):
    match = re.search(r"_(\d{10})_(\d{10})\.wav$", audio_path.name, flags=re.IGNORECASE)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if end > start:
            return (end - start) / 32000.0
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if frame_rate > 0:
            return frame_count / float(frame_rate)
    except Exception:
        return None
    return None


def reference_row_is_valid(audio_path, prompt_text):
    prompt_text = prompt_text.strip()
    if not audio_path.exists() or not prompt_text:
        return False
    duration_seconds = get_reference_duration_seconds(audio_path)
    if duration_seconds is None:
        return True
    return REFERENCE_MIN_SECONDS <= duration_seconds <= REFERENCE_MAX_SECONDS


def parse_reference_from_list(character_name, list_path):
    slicer_dir = GPT_ROOT / "output" / "slicer_opt" / character_name
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        audio_field, _, _, prompt_text = parts
        audio_path = Path(audio_field)
        if not audio_path.is_absolute():
            gpt_relative = GPT_ROOT / audio_path
            if gpt_relative.exists():
                audio_path = gpt_relative
            else:
                audio_path = slicer_dir / audio_path.name
        if reference_row_is_valid(audio_path, prompt_text):
            return audio_path.resolve(), prompt_text.strip()
    raise RuntimeError(
        "Could not find a valid reference row in the corrected list "
        f"(requires existing audio, non-empty text, and duration between "
        f"{REFERENCE_MIN_SECONDS:.0f}-{REFERENCE_MAX_SECONDS:.0f} seconds)"
    )


def detect_lang(text, preferred=None):
    preferred = normalize_language(preferred)
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "ja"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        if preferred in {"zh", "yue"}:
            return preferred
        return "zh"
    return "en"


def infer_language_from_list(list_path, fallback="ja"):
    scores = {}
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        list_language = normalize_language(parts[2])
        prompt_text = parts[3].strip()
        if list_language and list_language != "auto":
            scores[list_language] = scores.get(list_language, 0) + 3
        if prompt_text:
            detected = detect_lang(prompt_text, preferred=list_language)
            scores[detected] = scores.get(detected, 0) + 1
    if not scores:
        return normalize_language(fallback, "ja")
    priority = {"yue": 4, "zh": 3, "ja": 2, "en": 1}
    return max(scores.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]


def resolve_requested_language(args, cfg):
    override = normalize_language(getattr(args, "language", None))
    if override:
        return override
    return normalize_language(cfg["gpt_sovits"].get("language"), "ja")


def write_session(character_name, list_path, cfg):
    version = cfg["gpt_sovits"]["version"]
    gpt_weights = find_latest_weight(GPT_ROOT / f"GPT_weights_{version}", f"{character_name}-*.ckpt")
    sovits_weights = find_latest_weight(GPT_ROOT / f"SoVITS_weights_{version}", f"{character_name}_*.pth")
    ref_audio, prompt_text = parse_reference_from_list(character_name, list_path)
    session_language = normalize_language(cfg["gpt_sovits"].get("language"), "ja")
    session = {
        "character": character_name,
        "list_path": str(list_path),
        "gpt_weights": str(gpt_weights),
        "sovits_weights": str(sovits_weights),
        "reference_audio": str(ref_audio),
        "prompt_text": prompt_text,
        "prompt_lang": detect_lang(prompt_text, preferred=session_language),
        "text_lang": normalize_language(cfg["bridge"].get("text_lang"), session_language) or session_language,
        "language": session_language,
        "version": version,
    }
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Session written: {SESSION_PATH}")
    return session


def api_base_url(cfg):
    if using_remote_gpt(cfg):
        base = cfg["gpt_sovits"].get("remote_api_base", "").strip().rstrip("/")
        if not base or base == "https://example.com":
            raise ValueError("Remote GPT-SoVITS mode requires a real gpt_sovits.remote_api_base in config/defaults.json")
        return base
    return f"http://{cfg['gpt_sovits']['api_host']}:{cfg['gpt_sovits']['api_port']}"


def wait_for_api(cfg, timeout_seconds=60):
    url = api_base_url(cfg) + "/tts"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            request = urllib.request.Request(url, method="GET")
            urllib.request.urlopen(request, timeout=2)
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError("GPT-SoVITS API did not become ready in time")


def set_api_weight(cfg, endpoint, weight_path):
    if using_remote_gpt(cfg) and not cfg["gpt_sovits"].get("manage_remote_weights", True):
        print(f"Skipping {endpoint}: remote weight management disabled in config")
        return
    resolved_weight_path = Path(remap_to_current_gpt_root(weight_path))
    relative = resolved_weight_path.relative_to(GPT_ROOT).as_posix()
    query = urllib.parse.urlencode({"weights_path": relative})
    url = f"{api_base_url(cfg)}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    print(f"{endpoint}: {body}")


def start_api(cfg):
    if using_remote_gpt(cfg):
        print(f"Using remote GPT-SoVITS API: {api_base_url(cfg)}")
        return None
    existing_pid = find_listener_pid(cfg["gpt_sovits"]["api_port"])
    if existing_pid:
        print(
            f"Reusing existing GPT-SoVITS API on "
            f"{cfg['gpt_sovits']['api_host']}:{cfg['gpt_sovits']['api_port']} (PID {existing_pid})"
        )
        pids = load_pids()
        pids["gpt_api"] = existing_pid
        save_pids(pids)
        return None

    runtime_python = build_runtime_python(windowless=True)
    command = [
        str(runtime_python),
        "api_v2.py",
        "-a",
        cfg["gpt_sovits"]["api_host"],
        "-p",
        str(cfg["gpt_sovits"]["api_port"]),
        "-c",
        "GPT_SoVITS/configs/tts_infer.yaml",
    ]
    stdout_log = LOG_DIR / "gpt_api_stdout.log"
    stderr_log = LOG_DIR / "gpt_api_stderr.log"
    process = start_detached(command, GPT_ROOT, stdout_path=stdout_log, stderr_path=stderr_log)
    record_pid("gpt_api", process)
    print(f"GPT-SoVITS API logs: {stdout_log} / {stderr_log}")
    return process


def build_vtuber_command(session, cfg):
    easy_python = build_easy_python(windowless=True)
    ensure_exists(easy_python, "EasyVtuber python")
    command = [
        str(easy_python),
        "-m",
        "src.main",
        "--text_input",
        "--character",
        session["character"],
        "--alpha_clean",
        "--simplify",
        "3",
        "--cache",
        cfg["easyvtuber"]["cache"],
        "--gpu_cache",
        cfg["easyvtuber"]["gpu_cache"],
        "--use_interpolation",
        "--interpolation_half",
        "--interpolation_scale",
        "4",
        "--model_version",
        "v3",
        "--frame_rate_limit",
        cfg["easyvtuber"]["frame_rate_limit"],
        "--filter_min_cutoff",
        cfg["easyvtuber"]["filter_min_cutoff"],
        "--filter_beta",
        cfg["easyvtuber"]["filter_beta"],
    ]
    if cfg["easyvtuber"].get("use_sr"):
        command.append("--use_sr")
    if cfg["easyvtuber"].get("sr_half"):
        command.append("--sr_half")
    if cfg["easyvtuber"].get("sr_x4"):
        command.append("--sr_x4")
    if cfg["easyvtuber"].get("sr_a4k"):
        command.append("--sr_a4k")
    output_mode = cfg["easyvtuber"].get("output_mode", "debug")
    if output_mode == "desktop_pet":
        command.append("--output_desktop_pet")
    else:
        command.append("--output_debug")
    return command


def start_vtuber(session, cfg):
    env = os.environ.copy()
    env["EZVTB_DEVICE_ID"] = cfg["easyvtuber"]["device_id"]
    stdout_log = LOG_DIR / "easyvtuber_stdout.log"
    stderr_log = LOG_DIR / "easyvtuber_stderr.log"
    process = start_detached(
        build_vtuber_command(session, cfg),
        EASY_ROOT,
        env=env,
        stdout_path=stdout_log,
        stderr_path=stderr_log,
    )
    record_pid("easyvtuber", process)
    print(f"EasyVtuber logs: {stdout_log} / {stderr_log}")
    return process


def build_bridge_command(session):
    easy_python = build_easy_python()
    cfg = load_config()
    command = [
        str(easy_python),
        "gptsovits_bridge.py",
        "--interactive",
        "--api-base",
        api_base_url(cfg),
        "--character",
        session["character"],
        "--ref-audio-path",
        session["reference_audio"],
        "--prompt-text",
        session["prompt_text"],
        "--prompt-lang",
        session["prompt_lang"],
        "--text-lang",
        session.get("text_lang", session.get("language", "ja")),
    ]
    if cfg["bridge"].get("llm"):
        command.append("--llm")
    return command


def run_bridge(session):
    run_checked(build_bridge_command(session), EASY_ROOT)


def cmd_prepare(args, cfg):
    character_name = choose_character_name(args)
    requested_language = resolve_requested_language(args, cfg)
    active_cfg = config_with_language(cfg, requested_language)
    copy_character_image(character_name)
    audio_path = resolve_training_audio(character_name, args)
    list_path = slice_and_asr(character_name, active_cfg, audio_path=audio_path)
    if requested_language == "auto":
        detected_language = infer_language_from_list(list_path, fallback=cfg["gpt_sovits"]["language"])
        print(f"Detected language from ASR output: {detected_language}")
        active_cfg = config_with_language(cfg, detected_language)
    review_list(list_path)
    return character_name, list_path, active_cfg


def cmd_train(args, cfg):
    character_name = choose_character_name(args)
    if should_remove_bg(args):
        print("remove-bg requested; regenerating slices/list before training.")
        character_name, list_path, active_cfg = cmd_prepare(args, cfg)
        train_character(character_name, list_path, active_cfg)
        return write_session(character_name, list_path, active_cfg)
    list_path = LIST_DIR / f"{character_name}.list"
    ensure_exists(list_path, "Corrected list")
    requested_language = resolve_requested_language(args, cfg)
    active_language = requested_language
    if requested_language == "auto":
        active_language = infer_language_from_list(list_path, fallback=cfg["gpt_sovits"]["language"])
        print(f"Detected language from corrected list: {active_language}")
    active_cfg = config_with_language(cfg, active_language)
    train_character(character_name, list_path, active_cfg)
    return write_session(character_name, list_path, active_cfg)


def cmd_full(args, cfg):
    character_name, list_path, active_cfg = cmd_prepare(args, cfg)
    train_character(character_name, list_path, active_cfg)
    write_session(character_name, list_path, active_cfg)


def cmd_use(args, cfg):
    character_name = choose_character_name(args)
    list_path = find_character_list(character_name)
    requested_language = normalize_language(getattr(args, "language", None))
    if requested_language in {None, "auto"}:
        active_language = infer_language_from_list(list_path, fallback=cfg["gpt_sovits"]["language"])
        print(f"Inferred session language for {character_name}: {active_language}")
    else:
        active_language = requested_language
    active_cfg = config_with_language(cfg, active_language)
    session = write_session(character_name, list_path, active_cfg)
    print(f"Switched current session to: {character_name}")
    return session


def cmd_launch(args, cfg):
    ensure_exists(SESSION_PATH, "Session file")
    session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    normalized_session = normalize_session_paths(session)
    if normalized_session != session:
        SESSION_PATH.write_text(json.dumps(normalized_session, ensure_ascii=False, indent=2), encoding="utf-8")
        session = normalized_session
        print(f"Normalized session paths to current GPT-SoVITS root: {SESSION_PATH}")
    copy_character_image(session["character"])
    start_api(cfg)
    wait_for_api(cfg)
    set_api_weight(cfg, "set_gpt_weights", session["gpt_weights"])
    set_api_weight(cfg, "set_sovits_weights", session["sovits_weights"])
    start_vtuber(session, cfg)
    if args.interactive:
        try:
            run_bridge(session)
        except KeyboardInterrupt:
            print("\nInterrupted. Stopping launch processes...")
        finally:
            stop_tracked_processes()
    else:
        print("Run this to enter TTS interaction:")
        print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in build_bridge_command(session)))


def build_parser():
    parser = argparse.ArgumentParser(description="Thin one-click release workflow.")
    parser.add_argument("--character", help="Override the character name. Defaults to the input image stem.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_character_option(subparser):
        subparser.add_argument(
            "--character",
            help="Override the character name. Defaults to the input image stem.",
        )

    def add_language_option(subparser):
        subparser.add_argument(
            "--language",
            choices=sorted(SUPPORTED_LANGUAGES),
            help="Training/ASR language override. Use auto to infer from ASR output.",
        )

    def add_remove_option(subparser):
        subparser.add_argument(
            "--remove-bg",
            action="store_true",
            help="Run background/accompaniment removal before slicing and training.",
        )
        subparser.add_argument(
            "--remove",
            choices=["bg"],
            help="Alias for --remove-bg. Example: --remove bg",
        )

    prepare = subparsers.add_parser("prepare", help="Slice audio, ASR it, open the generated list for correction.")
    add_character_option(prepare)
    add_language_option(prepare)
    add_remove_option(prepare)

    train = subparsers.add_parser("train", help="Train from the corrected list and write session metadata.")
    add_character_option(train)
    add_language_option(train)
    add_remove_option(train)

    full = subparsers.add_parser("full", help="Prepare, pause for list correction, then train.")
    add_character_option(full)
    add_language_option(full)
    add_remove_option(full)

    use = subparsers.add_parser("use", help="Switch the current session to an existing trained character.")
    add_character_option(use)
    add_language_option(use)

    launch = subparsers.add_parser("launch", help="Start API, switch weights, start VTuber, optionally enter TTS.")
    add_character_option(launch)
    launch.add_argument("--interactive", action="store_true", help="Run the interactive bridge in the current console.")
    subparsers.add_parser("stop", help="Stop tracked background API/VTuber processes.")
    subparsers.add_parser("status", help="Show tracked background API/VTuber processes.")
    return parser


def main():
    cfg = load_config()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args, cfg)
    elif args.command == "train":
        cmd_train(args, cfg)
    elif args.command == "full":
        cmd_full(args, cfg)
    elif args.command == "use":
        cmd_use(args, cfg)
    elif args.command == "launch":
        cmd_launch(args, cfg)
    elif args.command == "stop":
        stop_tracked_processes()
    elif args.command == "status":
        print_status()


if __name__ == "__main__":
    main()
