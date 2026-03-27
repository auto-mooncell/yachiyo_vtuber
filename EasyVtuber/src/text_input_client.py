import json
import os
import time
from multiprocessing import Process, Value, shared_memory

import numpy as np

from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.timer_wait import wait_until

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MOTION_PATH = os.path.join(PROJECT_ROOT, "text_motion.json")
MOUTH_PATH = os.path.join(PROJECT_ROOT, "mouth.txt")
MOUTH_SCAN_SEQUENCE = [
    ("mouth_aaa", {26: 1.0}),
    ("mouth_iii", {27: 1.0}),
    ("mouth_uuu", {28: 1.0}),
    ("mouth_eee", {29: 1.0}),
    ("mouth_ooo", {30: 1.0}),
    ("mouth_delta", {31: 1.0}),
    ("mouth_aaa+delta", {26: 1.0, 31: 1.0}),
    ("mouth_lowered_corners", {32: 1.0, 33: 1.0}),
    ("mouth_raised_corners", {34: 1.0, 35: 1.0}),
    ("mouth_smirk", {36: 1.0}),
]


def _clamp(value, low, high):
    return max(low, min(high, value))


def _read_mouth_open():
    try:
        with open(TEXT_MOTION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "mouth_open" in data:
            return _clamp(float(data["mouth_open"]), 0.0, 1.0)
    except Exception:
        pass

    try:
        with open(MOUTH_PATH, "r", encoding="utf-8") as f:
            return _clamp(float(f.read().strip()), 0.0, 1.0)
    except Exception:
        return 0.0


def _read_text_control():
    control = {
        "mouth_open": 0.0,
        "mouth_smile": 0.0,
        "eyebrow_raise": 0.0,
        "eyebrow_happy": 0.0,
        "body_x": 0.0,
        "body_y": 0.0,
        "body_scale": 0.0,
        "body_rotation": 0.0,
    }

    try:
        with open(TEXT_MOTION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, low, high in [
            ("mouth_open", 0.0, 1.0),
            ("mouth_smile", 0.0, 1.0),
            ("eyebrow_raise", 0.0, 1.0),
            ("eyebrow_happy", 0.0, 1.0),
            ("body_x", -1.0, 1.0),
            ("body_y", -1.0, 1.0),
            ("body_scale", -1.0, 1.0),
            ("body_rotation", -1.0, 1.0),
        ]:
            if key in data:
                control[key] = _clamp(float(data[key]), low, high)
        return control
    except Exception:
        pass

    control["mouth_open"] = _read_mouth_open()
    control["mouth_smile"] = control["mouth_open"] * 0.25
    control["eyebrow_raise"] = control["mouth_open"] * 0.08
    control["eyebrow_happy"] = control["mouth_open"] * 0.05
    return control


def _get_test_mouth_open():
    phase = int(time.perf_counter() / 2.0) % 2
    return 1.0 if phase == 1 else 0.0


def _get_scan_pose():
    step_index = int(time.perf_counter() / 2.0) % len(MOUTH_SCAN_SEQUENCE)
    label, values = MOUTH_SCAN_SEQUENCE[step_index]
    pose = np.zeros((45,), dtype=np.float32)
    for index, value in values.items():
        pose[index] = value
    return label, pose


class TextInputClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value("f", 60.0)

    def run(self):
        last_time = time.perf_counter()
        interval = 1.0 / 60.0
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[: 45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4 : 45 * 4 + 4 * 4])
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        last_scan_label = None

        print("Text Input Running at 60 FPS")
        while True:
            if args.mouth_scan:
                scan_label, model_input_arr = _get_scan_pose()
                if scan_label != last_scan_label:
                    print("Testing mouth channel:", scan_label)
                    last_scan_label = scan_label
            else:
                model_input_arr = np.zeros((45,), dtype=np.float32)
                if args.text_input_test:
                    control = {
                        "mouth_open": _get_test_mouth_open(),
                        "mouth_smile": 0.15,
                        "eyebrow_raise": 0.12,
                        "eyebrow_happy": 0.08,
                        "body_x": 0.15,
                        "body_y": -0.2,
                        "body_scale": 0.05,
                        "body_rotation": 0.12,
                    }
                else:
                    control = _read_text_control()

                base_mouth_open = control["mouth_open"]
                mouth_smile = control["mouth_smile"]
                t = time.perf_counter()
                chatter_phase = t * 12.0
                chatter = 0.5 + 0.5 * np.sin(chatter_phase)
                mouth_open = _clamp(base_mouth_open * (0.25 + 0.75 * chatter), 0.0, 1.0)
                phrase_turn = np.sin(t * 1.9)
                phrase_turn_fast = np.sin(t * 3.6 + 0.8)
                phrase_tilt = np.sin(t * 2.5 + 1.1)
                nod = np.sin(t * 5.1)
                idle_strength = 0.34
                emphasis = _clamp(idle_strength + mouth_open * 0.78, 0.0, 1.0)
                eyebrow_raise = _clamp(control["eyebrow_raise"] + mouth_open * 0.10, 0.0, 1.0)
                eyebrow_happy = _clamp(control["eyebrow_happy"] + mouth_smile * 0.65, 0.0, 1.0)

                # Mouth
                model_input_arr[26] = mouth_open
                model_input_arr[27] = mouth_open * 0.18
                model_input_arr[30] = mouth_open * 0.42
                model_input_arr[31] = mouth_open * 0.55
                model_input_arr[34] = min(1.0, mouth_smile + mouth_open * 0.15)
                model_input_arr[35] = min(1.0, mouth_smile + mouth_open * 0.15)
                model_input_arr[36] = _clamp(0.08 * mouth_open * max(0.0, phrase_turn_fast), 0.0, 1.0)

                # Brows. These only take effect when eyebrow mode is enabled in the launcher.
                model_input_arr[6] = eyebrow_raise * 0.92
                model_input_arr[7] = eyebrow_raise * 0.84
                model_input_arr[8] = eyebrow_happy * 0.86
                model_input_arr[9] = eyebrow_happy * 0.94

                # Head / neck / body / breath. Keep it layered rather than shaky.
                model_input_arr[39] = _clamp(
                    -control["body_y"] * 0.40 + nod * 0.08 * emphasis,
                    -1.0,
                    1.0,
                )
                model_input_arr[40] = _clamp(
                    control["body_x"] * 0.38 + phrase_turn * 0.14 * emphasis + phrase_turn_fast * 0.05 * emphasis,
                    -1.0,
                    1.0,
                )
                model_input_arr[41] = _clamp(
                    control["body_rotation"] * 0.36 + phrase_tilt * 0.12 * emphasis,
                    -1.0,
                    1.0,
                )
                model_input_arr[42] = _clamp(
                    control["body_x"] * 0.28 + phrase_turn * 0.16 * emphasis,
                    -1.0,
                    1.0,
                )
                model_input_arr[43] = _clamp(
                    control["body_rotation"] * 0.28 + phrase_tilt * 0.14 * emphasis,
                    -1.0,
                    1.0,
                )
                model_input_arr[44] = _clamp(
                    0.12 + mouth_open * 0.18 + control["body_scale"] * 0.20,
                    0.0,
                    1.0,
                )

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                np_position_shm[:] = position_vector

            wait_until(last_time + interval)
            last_time += interval
