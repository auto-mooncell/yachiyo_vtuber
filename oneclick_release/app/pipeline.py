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
IMAGE_INPUT_DIR = RELEASE_ROOT / "data" / "input" / "image"
AUDIO_INPUT_DIR = RELEASE_ROOT / "data" / "input" / "audio"
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 10.0

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


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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


def choose_character_name(args):
    if args.character:
        return args.character
    image_path = find_single_file(IMAGE_INPUT_DIR, {".png"})
    return image_path.stem


def copy_character_image(character_name):
    exact_match = IMAGE_INPUT_DIR / f"{character_name}.png"
    if exact_match.exists():
        image_path = exact_match
    else:
        image_path = find_single_file(IMAGE_INPUT_DIR, {".png"})
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


def slice_and_asr(character_name, cfg):
    audio_path = find_single_file(AUDIO_INPUT_DIR, {".wav", ".mp3", ".flac", ".m4a", ".ogg"})
    slicer_dir = GPT_ROOT / "output" / "slicer_opt" / character_name
    asr_dir = GPT_ROOT / "output" / "asr_opt"
    runtime_python = build_runtime_python()
    ensure_exists(runtime_python, "GPT-SoVITS runtime python")

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
    run_checked(command, GPT_ROOT)

    source_list = asr_dir / f"{character_name}.list"
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


def detect_lang(text):
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "ja"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


def write_session(character_name, list_path, cfg):
    version = cfg["gpt_sovits"]["version"]
    gpt_weights = find_latest_weight(GPT_ROOT / f"GPT_weights_{version}", f"{character_name}-*.ckpt")
    sovits_weights = find_latest_weight(GPT_ROOT / f"SoVITS_weights_{version}", f"{character_name}_*.pth")
    ref_audio, prompt_text = parse_reference_from_list(character_name, list_path)
    session = {
        "character": character_name,
        "list_path": str(list_path),
        "gpt_weights": str(gpt_weights),
        "sovits_weights": str(sovits_weights),
        "reference_audio": str(ref_audio),
        "prompt_text": prompt_text,
        "prompt_lang": detect_lang(prompt_text),
        "text_lang": cfg["bridge"]["text_lang"],
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
    relative = Path(weight_path).relative_to(GPT_ROOT).as_posix()
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
        session["text_lang"],
    ]
    if cfg["bridge"].get("llm"):
        command.append("--llm")
    return command


def run_bridge(session):
    run_checked(build_bridge_command(session), EASY_ROOT)


def cmd_prepare(args, cfg):
    character_name = choose_character_name(args)
    copy_character_image(character_name)
    list_path = slice_and_asr(character_name, cfg)
    review_list(list_path)
    return character_name, list_path


def cmd_train(args, cfg):
    character_name = choose_character_name(args)
    list_path = LIST_DIR / f"{character_name}.list"
    ensure_exists(list_path, "Corrected list")
    train_character(character_name, list_path, cfg)
    return write_session(character_name, list_path, cfg)


def cmd_full(args, cfg):
    character_name, list_path = cmd_prepare(args, cfg)
    train_character(character_name, list_path, cfg)
    write_session(character_name, list_path, cfg)


def cmd_use(args, cfg):
    character_name = choose_character_name(args)
    list_path = find_character_list(character_name)
    session = write_session(character_name, list_path, cfg)
    print(f"Switched current session to: {character_name}")
    return session


def cmd_launch(args, cfg):
    ensure_exists(SESSION_PATH, "Session file")
    session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
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

    prepare = subparsers.add_parser("prepare", help="Slice audio, ASR it, open the generated list for correction.")
    add_character_option(prepare)

    train = subparsers.add_parser("train", help="Train from the corrected list and write session metadata.")
    add_character_option(train)

    full = subparsers.add_parser("full", help="Prepare, pause for list correction, then train.")
    add_character_option(full)

    use = subparsers.add_parser("use", help="Switch the current session to an existing trained character.")
    add_character_option(use)

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
