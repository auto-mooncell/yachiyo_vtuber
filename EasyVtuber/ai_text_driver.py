import argparse
import json
import math
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def clamp(value, low, high):
    return max(low, min(high, value))


def neutral_state():
    return {
        "mouth_open": 0.0,
        "mouth_smile": 0.0,
        "body_x": 0.0,
        "body_y": 0.0,
        "body_scale": 0.0,
        "body_rotation": 0.0,
    }


def char_profile(ch):
    strong_pause = "\u3002\uff01\uff1f!?"
    weak_pause = "\uff0c\u3001\uff1b\uff1a,;:"

    if not ch:
        return {"duration": 0.12, "mouth": 0.0, "smile": 0.0, "energy": 0.0, "pause": True}
    if ch.isspace():
        return {"duration": 0.10, "mouth": 0.0, "smile": 0.0, "energy": 0.0, "pause": True}
    if ch in strong_pause:
        return {"duration": 0.30, "mouth": 0.0, "smile": 0.0, "energy": 0.0, "pause": True}
    if ch in weak_pause:
        return {"duration": 0.16, "mouth": 0.0, "smile": 0.0, "energy": 0.0, "pause": True}
    if "\u4e00" <= ch <= "\u9fff":
        return {"duration": 0.16, "mouth": 1.00, "smile": 0.08, "energy": 0.85, "pause": False}

    lower = ch.lower()
    if lower in "aeo":
        return {"duration": 0.14, "mouth": 1.00, "smile": 0.08, "energy": 1.00, "pause": False}
    if lower in "iuvwy":
        return {"duration": 0.12, "mouth": 0.70, "smile": 0.10, "energy": 0.70, "pause": False}
    if lower in "mbp":
        return {"duration": 0.09, "mouth": 0.28, "smile": 0.02, "energy": 0.45, "pause": False}
    return {"duration": 0.12, "mouth": 0.72, "smile": 0.06, "energy": 0.65, "pause": False}


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


def animate_text(text, fps, chars_per_second, motion_path, mouth_path):
    if not text.strip():
        write_motion(neutral_state(), motion_path, mouth_path)
        return

    base_duration = 1.0 / max(chars_per_second, 0.1)

    for index, ch in enumerate(text):
        profile = char_profile(ch)
        duration = max(profile["duration"], base_duration)
        if profile["pause"]:
            write_motion(neutral_state(), motion_path, mouth_path)
            time.sleep(duration)
            continue

        attack = min(0.12, duration * 0.25)
        hold = max(0.35, duration * 0.55)
        release = min(0.18, max(duration - attack - hold, 0.08))
        sway = -1.0 if index % 2 == 0 else 1.0
        energy = profile["energy"]

        open_state = {
            "mouth_open": clamp(profile["mouth"], 0.0, 1.0),
            "mouth_smile": clamp(profile["smile"], 0.0, 1.0),
            "body_x": clamp(0.18 * sway * energy, -1.0, 1.0),
            "body_y": clamp(-0.24 * energy, -1.0, 1.0),
            "body_scale": clamp(0.10 * energy, -1.0, 1.0),
            "body_rotation": clamp(0.14 * sway * energy, -1.0, 1.0),
        }

        half_open_state = {
            "mouth_open": clamp(profile["mouth"] * 0.55, 0.0, 1.0),
            "mouth_smile": clamp(profile["smile"] * 0.8, 0.0, 1.0),
            "body_x": clamp(0.10 * sway * energy, -1.0, 1.0),
            "body_y": clamp(-0.12 * energy, -1.0, 1.0),
            "body_scale": clamp(0.05 * energy, -1.0, 1.0),
            "body_rotation": clamp(0.07 * sway * energy, -1.0, 1.0),
        }

        write_motion(half_open_state, motion_path, mouth_path)
        time.sleep(attack)
        write_motion(open_state, motion_path, mouth_path)
        time.sleep(hold)
        write_motion(half_open_state, motion_path, mouth_path)
        time.sleep(release)

    write_motion(neutral_state(), motion_path, mouth_path)
    time.sleep(max(0.25, 1.0 / max(fps, 0.1)))


def load_text(args):
    if args.text is not None:
        return args.text
    file_path = args.file
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(description="Drive EasyVtuber mouth/body motion from fixed text.")
    parser.add_argument("--text", type=str, help="Text to animate once or in a loop.")
    parser.add_argument("--file", type=str, default=os.path.join(PROJECT_ROOT, "text_input.txt"), help="Read text from a file when --text is not given.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chars-per-second", type=float, default=8.0)
    parser.add_argument("--loop", action="store_true", help="Loop the same text forever.")
    parser.add_argument("--motion-path", type=str, default=os.path.join(PROJECT_ROOT, "text_motion.json"))
    parser.add_argument("--mouth-path", type=str, default=os.path.join(PROJECT_ROOT, "mouth.txt"))
    args = parser.parse_args()

    text = load_text(args)
    print(f"Text motion driver started. Text length={len(text)}")
    print(f"Writing motion to {args.motion_path} and mouth fallback to {args.mouth_path}")

    try:
        while True:
            animate_text(
                text=text,
                fps=args.fps,
                chars_per_second=args.chars_per_second,
                motion_path=args.motion_path,
                mouth_path=args.mouth_path,
            )
            if not args.loop:
                break
    finally:
        write_motion(neutral_state(), args.motion_path, args.mouth_path)


if __name__ == "__main__":
    main()
