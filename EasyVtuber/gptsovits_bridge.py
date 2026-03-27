import argparse
import glob
import io
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from dotenv import load_dotenv

# Optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHARACTER_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "character_prompts.json")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def clean_env_value(value):
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def load_character_prompt(character_name):
    default_prompt = (
        "You are a roleplay assistant for a VTuber character. "
        "Stay in character and always return exactly one JSON object with the keys "
        "text, emotion, action."
    )
    try:
        with open(CHARACTER_PROMPTS_PATH, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except Exception:
        return default_prompt

    if character_name:
        normalized = character_name.strip()
        for key in (normalized, normalized.lower()):
            if key in prompts:
                return prompts[key]
        for key, value in prompts.items():
            if key.lower() == normalized.lower():
                return value
    return prompts.get("default", default_prompt)


def detect_lang(text):
    """Simple language detection based on character sets."""
    if not text:
        return "zh"
    # Japanese characters (Hiragana/Katakana)
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "ja"
    # Chinese characters (Simplified/Traditional)
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return "zh"
    # English/Latin
    if any(("a" <= c <= "z" or "A" <= c <= "Z") for c in text):
        return "en"
    return "zh"


class ChatAgent:
    def __init__(self, api_key, base_url, model="gpt-4o"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = (
            "你是弥千世（Yachiyo），一个拥有8000年知识和经验的博学少女。你说话语气温和、充满智慧，偶尔带点俏皮。"
            "你的输出必须是合法的JSON格式，包含以下字段：\n"
            "- text: 你的回复内容（必须使用用户使用的语种）。\n"
            "- emotion: 你的心情，可选值：neutral, happy, sad, angry, surprised。\n"
            "- action: 你的动作，可选值：none, nod, shake, tilt。\n"
            "不要输出任何JSON以外的内容。"
        )
        self.history = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            self.history.append({"role": "assistant", "content": content})
            return data
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return {"text": f"抱歉，我走神了（{e}）", "emotion": "neutral", "action": "none"}


class CharacterPromptChatAgent(ChatAgent):
    def __init__(self, api_key, base_url, model="gpt-4o", character_name=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = load_character_prompt(character_name)
        self.history = [{"role": "system", "content": self.system_prompt}]


ChatAgent = CharacterPromptChatAgent


def clamp(value, low, high):
    return max(low, min(high, value))


def neutral_state():
    return {
        "mouth_open": 0.0,
        "mouth_smile": 0.0,
        "eyebrow_raise": 0.0,
        "eyebrow_happy": 0.0,
        "body_x": 0.0,
        "body_y": 0.0,
        "body_scale": 0.0,
        "body_rotation": 0.0,
    }


def write_motion(state, motion_path, mouth_path):
    tmp_path = motion_path + ".tmp"
    payload = json.dumps(state, ensure_ascii=False)

    replaced = False
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, motion_path)
        replaced = True
    except PermissionError:
        pass
    finally:
        if not replaced and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not replaced:
        for _ in range(3):
            try:
                with open(motion_path, "w", encoding="utf-8") as f:
                    f.write(payload)
                replaced = True
                break
            except PermissionError:
                time.sleep(0.01)

    with open(mouth_path, "w", encoding="utf-8") as f:
        f.write(f"{state['mouth_open']:.4f}")


def auto_resolve_ref():
    """Try to find Yachiyo/弥千世 files first, then fallback to the first .wav in default dir."""
    search_dir = os.path.join(PROJECT_ROOT, "..", "GPT-SoVITS-v2pro-20250604", "GPT_weights_v2Pro")
    if not os.path.exists(search_dir):
        return None, "", "zh"

    wavs = glob.glob(os.path.join(search_dir, "*.wav")) + glob.glob(os.path.join(search_dir, "*.WAV"))
    if not wavs:
        return None, "", "zh"

    # Prioritize files containing 'Yachiyo' or '弥千世'
    yachiyo_wavs = [w for w in wavs if "yachiyo" in w.lower() or "弥千世" in w]
    if yachiyo_wavs:
        yachiyo_wavs.sort()
        ref_audio = yachiyo_wavs[0]
    else:
        wavs.sort()
        ref_audio = wavs[0]

    print(f"Selecting Reference Audio: {os.path.basename(ref_audio)}")
    prompt_text = ""

    # Try to find a .txt with the same name
    txt_path = os.path.splitext(ref_audio)[0] + ".txt"
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read prompt text from {txt_path}: {e}")

    prompt_lang = detect_lang(prompt_text)
    return ref_audio, prompt_text, prompt_lang


def fetch_tts_wav(api_base, payload):
    url = api_base.rstrip("/") + "/tts"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            return body, content_type
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPT-SoVITS API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GPT-SoVITS API unavailable: {e}") from e


def decode_wav_bytes(wav_bytes):
    rate, audio = wavfile.read(io.BytesIO(wav_bytes))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    return rate, audio


class AudioMotionPlayer:
    def __init__(self, audio, sample_rate, motion_path, mouth_path, emotion="neutral", action="none"):
        if audio.ndim == 1:
            audio = audio[:, None]
        self.audio = audio
        self.sample_rate = sample_rate
        self.motion_path = motion_path
        self.mouth_path = mouth_path
        self.channels = audio.shape[1]
        self.position = 0
        self.last_level = 0.0
        self.finished = False
        self.lock = threading.Lock()
        
        # Emotion and Action states
        self.emotion = emotion
        self.action = action
        self.play_start_time = 0

    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status)

        end = min(self.position + frames, len(self.audio))
        chunk = self.audio[self.position:end]
        outdata.fill(0)
        if len(chunk) > 0:
            outdata[: len(chunk)] = chunk
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float32)))
        else:
            rms = 0.0

        with self.lock:
            self.position = end
            self.last_level = rms
            self.finished = end >= len(self.audio)

        if end >= len(self.audio):
            raise sd.CallbackStop()

    def motion_state(self, smoothed_level, elapsed):
        speak = clamp(smoothed_level * 4.2, 0.0, 1.0)
        
        sway = math.sin(elapsed * 1.9)
        sway_fast = math.sin(elapsed * 3.7 + 0.8)
        nod_osc = math.sin(elapsed * 5.2)
        tilt_osc = math.sin(elapsed * 2.6 + 1.4)
        
        idle = 0.30
        energy = clamp(idle + speak * 0.82, 0.0, 1.0)

        # Emotion mappings
        e_smile = 0.0
        e_eyebrow = 0.0
        if self.emotion == "happy":
            e_smile = 0.6
            e_eyebrow = 0.3
        elif self.emotion == "sad":
            e_smile = -0.2
            e_eyebrow = -0.4
        elif self.emotion == "angry":
            e_smile = -0.3
            e_eyebrow = 0.5
        elif self.emotion == "surprised":
            e_smile = 0.0
            e_eyebrow = 0.7

        # Action logic (one-time triggered in first 1.5s)
        action_x = 0.0
        action_y = 0.0
        action_rot = 0.0
        if elapsed < 1.5:
            if self.action == "nod":
                action_y = -0.25 * math.sin(elapsed * 4.0)
            elif self.action == "shake":
                action_x = 0.25 * math.sin(elapsed * 6.0)
            elif self.action == "tilt":
                action_rot = 0.2 * math.sin(elapsed * 3.0)

        return {
            "mouth_open": clamp(0.12 + speak * 1.15, 0.0, 1.0),
            "mouth_smile": clamp(0.04 + speak * 0.12 + e_smile, 0.0, 1.0),
            "eyebrow_raise": clamp(0.05 + speak * 0.16 + max(0.0, nod_osc) * 0.10 + e_eyebrow, 0.0, 1.0),
            "eyebrow_happy": clamp(0.03 + speak * 0.14 + (e_smile * 0.5), 0.0, 1.0),
            "body_x": clamp((sway * 0.22 + sway_fast * 0.07 + action_x) * energy, -1.0, 1.0),
            "body_y": clamp(-0.16 * energy + nod_osc * 0.08 * energy + action_y, -1.0, 1.0),
            "body_scale": clamp(0.08 * energy, -1.0, 1.0),
            "body_rotation": clamp((tilt_osc * 0.18 + sway * 0.05 + action_rot) * energy, -1.0, 1.0),
        }

    def play(self):
        smoothed_level = 0.0
        self.play_start_time = time.perf_counter()
        write_motion(neutral_state(), self.motion_path, self.mouth_path)

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self.callback,
            blocksize=2048,
        ) as stream:
            while stream.active:
                with self.lock:
                    level = self.last_level
                    finished = self.finished
                smoothed_level = smoothed_level * 0.72 + level * 0.28
                elapsed = time.perf_counter() - self.play_start_time
                write_motion(self.motion_state(smoothed_level, elapsed), self.motion_path, self.mouth_path)
                if finished:
                    break
                time.sleep(1.0 / 30.0)

        write_motion(neutral_state(), self.motion_path, self.mouth_path)


def synthesize_and_play(args, text, utterance_index, emotion="neutral", action="none"):
    actual_text_lang = args.text_lang
    if actual_text_lang == "auto" or not actual_text_lang:
        actual_text_lang = detect_lang(text)
    
    actual_prompt_lang = args.prompt_lang
    if actual_prompt_lang == "auto" or not actual_prompt_lang:
        actual_prompt_lang = detect_lang(args.prompt_text)

    ref_audio_path = args.ref_audio_path
    if not os.path.isabs(ref_audio_path):
        ref_audio_path = os.path.abspath(ref_audio_path)

    payload = {
        "text": text,
        "text_lang": actual_text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": args.prompt_text,
        "prompt_lang": actual_prompt_lang,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "text_split_method": args.text_split_method,
        "speed_factor": args.speed_factor,
        "sample_steps": args.sample_steps,
        "media_type": "wav",
        "streaming_mode": False,
    }

    print(f"\n[TTS] Text: {text}")
    print(f"[TTS] Lang: {actual_text_lang}, Emotion: {emotion}, Action: {action}")
    print(f"[TTS] Requesting audio from: {args.api_base}")
    wav_bytes, content_type = fetch_tts_wav(args.api_base, payload)
    sample_rate, audio = decode_wav_bytes(wav_bytes)
    duration = len(audio) / float(sample_rate) if len(audio) else 0.0
    print(f"[TTS] Audio ready: {sample_rate} Hz, {duration:.2f}s")
    print("[TTS] Playing audio and driving motion...")

    player = AudioMotionPlayer(audio, sample_rate, args.motion_path, args.mouth_path, emotion, action)
    try:
        player.play()
    finally:
        write_motion(neutral_state(), args.motion_path, args.mouth_path)
    print("[TTS] Done.")


def set_backend_weights(api_base, gpt_path, sovits_path):
    """Tell the GPT-SoVITS API to load specific weights."""
    try:
        # Set GPT
        gpt_url = f"{api_base.rstrip('/')}/set_gpt_weights?weights_path={gpt_path}"
        with urllib.request.urlopen(gpt_url, timeout=30) as r:
            print(f"Backend GPT weight set to: {gpt_path}")
        
        # Set SoVITS
        sovits_url = f"{api_base.rstrip('/')}/set_sovits_weights?weights_path={sovits_path}"
        with urllib.request.urlopen(sovits_url, timeout=30) as r:
            print(f"Backend SoVITS weight set to: {sovits_path}")
    except Exception as e:
        print(f"Warning: Could not auto-set backend weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bridge GPT-SoVITS TTS audio to EasyVtuber motion.")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:9880")
    parser.add_argument("--text", type=str, help="Text to synthesize.")
    parser.add_argument("--interactive", action="store_true", help="Read text from stdin in a loop.")
    parser.add_argument("--llm", action="store_true", help="Enable OpenAI LLM interaction.")
    parser.add_argument("--character", type=str, help="Character name for loading the matching LLM persona prompt.")
    parser.add_argument("--openai-key", type=str, default=clean_env_value(os.getenv("OPENAI_API_KEY")))
    parser.add_argument("--openai-base", type=str, default=clean_env_value(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    parser.add_argument("--openai-model", type=str, default="gpt-4o")
    
    parser.add_argument("--text-lang", type=str, default="auto")
    parser.add_argument("--ref-audio-path", type=str)
    parser.add_argument("--prompt-text", type=str)
    parser.add_argument("--prompt-lang", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--text-split-method", type=str, default="cut5")
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--motion-path", type=str, default=os.path.join(PROJECT_ROOT, "text_motion.json"))
    parser.add_argument("--mouth-path", type=str, default=os.path.join(PROJECT_ROOT, "mouth.txt"))
    args = parser.parse_args()
    args.openai_key = clean_env_value(args.openai_key)
    args.openai_base = clean_env_value(args.openai_base) or "https://api.openai.com/v1"

    # Resolve TTS defaults
    if not args.ref_audio_path or not args.prompt_text:
        def_ref, def_prompt, def_lang = auto_resolve_ref()
        if not args.ref_audio_path and def_ref:
            args.ref_audio_path = def_ref
        if not args.prompt_text and def_prompt:
            args.prompt_text = def_prompt
            if args.prompt_lang == "auto":
                args.prompt_lang = def_lang

    # Initialize LLM Agent
    agent = None
    if args.llm:
        if not OPENAI_AVAILABLE:
            print("Error: openai library not installed. Run 'pip install openai'")
            return
        if not args.openai_key:
            print("Error: OpenAI API Key not provided. Set it in .env or use --openai-key")
            return
        agent = ChatAgent(args.openai_key, args.openai_base, args.openai_model, args.character)
        if args.character:
            print(f"LLM Mode Enabled. Using model: {args.openai_model}, character: {args.character}")
        else:
            print(f"LLM Mode Enabled. Using model: {args.openai_model}")

    # Start interaction
    try:
        utterance_index = 1
        while True:
            if args.text and utterance_index == 1:
                user_input = args.text
            else:
                try:
                    user_input = input("\nUser> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
            
            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                break

            if args.llm and agent:
                print("Thinking...")
                response_data = agent.chat(user_input)
                reply_text = response_data.get("text", "")
                emotion = response_data.get("emotion", "neutral")
                action = response_data.get("action", "none")
            else:
                reply_text = user_input
                emotion = "neutral"
                action = "none"

            synthesize_and_play(args, reply_text, utterance_index, emotion, action)
            utterance_index += 1
            
            if not args.interactive and not args.llm:
                break
    finally:
        write_motion(neutral_state(), args.motion_path, args.mouth_path)


if __name__ == "__main__":
    main()
