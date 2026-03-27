import math
import os
import json
from multiprocessing import Process, Value, shared_memory, Event
from typing import List

import cv2
import numpy as np
import time

from .ezvtb_rt_interface import get_core
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.pose_simplify import pose_simplify
from .utils.fps import FPS, Interval

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MOTION_PATH = os.path.join(PROJECT_ROOT, "text_motion.json")
MOUTH_PATH = os.path.join(PROJECT_ROOT, "mouth.txt")


def _clamp(value, low, high):
    return max(low, min(high, value))


def _read_text_motion():
    control = {
        "mouth_open": 0.0,
        "mouth_smile": 0.0,
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

    try:
        with open(MOUTH_PATH, "r", encoding="utf-8") as f:
            mouth_open = float(f.read().strip())
        control["mouth_open"] = _clamp(mouth_open, 0.0, 1.0)
        control["mouth_smile"] = control["mouth_open"] * 0.25
    except Exception:
        pass

    return control


def _apply_text_motion_to_model_input(model_input: np.ndarray, control: dict):
    # Match the project's existing mouth pipeline used by mouse/face clients.
    mouth_channel_index = 12 + 14
    model_input[mouth_channel_index] = _clamp(control["mouth_open"], 0.0, 1.0) * 1.8


def _composite_bgra_over_white(bgra_image: np.ndarray) -> np.ndarray:
    bgr = bgra_image[:, :, :3].astype(np.float32)
    alpha = (bgra_image[:, :, 3:4].astype(np.float32) / 255.0)
    white = np.full_like(bgr, 255.0)
    composed = bgr * alpha + white * (1.0 - alpha)
    return composed.astype(np.uint8)


class ModelClientProcess(Process):
    def __init__(self, input_image, pose_position_shm: shared_memory.SharedMemory, input_fps):
        super().__init__()
        self.input_image = input_image
        self.pose_position_shm = pose_position_shm  # 45 floats for pose, 4 floats for position

        self.alpha_width_scale = 2 if args.alpha_split else 1
        self.ret_channels = 3 if args.output_virtual_cam or args.output_debug else 4
        self.ret_shape = (
            args.interpolation_scale,
            args.model_output_size,
            self.alpha_width_scale * args.model_output_size,
            self.ret_channels,
        )
        self.ret_nbytes = (
            self.alpha_width_scale
            * args.interpolation_scale
            * args.model_output_size
            * args.model_output_size
            * self.ret_channels
        )
        self.ret_shared_mem = shared_memory.SharedMemory(create=True, size=self.ret_nbytes)

        self.last_model_interval = Value("f", 0.0)
        self.average_model_interval = Value("f", 0.0)
        self.cache_hit_ratio = Value("f", 0.0)
        self.gpu_cache_hit_ratio = Value("f", 0.0)
        self.pipeline_fps_number = Value("f", 0.0)
        self.output_pipeline_fps = Value("f", 0.0)
        self.input_fps = input_fps

        self.finish_event = Event()

    def run(self):
        if getattr(args, "mark_interpolated", False):
            os.environ["EZVTB_MARK_INTERPOLATED"] = "1"

        pose_position_shm_guard = SharedMemoryGuard(
            self.pose_position_shm,
            ctrl_name="pose_position_shm_ctrl",
        )
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[: 45 * 4])
        np_position_shm = np.ndarray(
            (4,),
            dtype=np.float32,
            buffer=self.pose_position_shm.buf[45 * 4 : 45 * 4 + 4 * 4],
        )

        ret_batch_shm_guard = [
            SharedMemoryGuard(self.ret_shared_mem, ctrl_name=f"ret_shm_ctrl_batch_{i}")
            for i in range(args.interpolation_scale)
        ]
        np_ret_shms = [
            np.ndarray(
                (
                    args.model_output_size,
                    self.alpha_width_scale * args.model_output_size,
                    self.ret_channels,
                ),
                dtype=np.uint8,
                buffer=self.ret_shared_mem.buf[
                    i
                    * self.alpha_width_scale
                    * args.model_output_size
                    * args.model_output_size
                    * self.ret_channels : (i + 1)
                    * self.alpha_width_scale
                    * args.model_output_size
                    * args.model_output_size
                    * self.ret_channels
                ],
            )
            for i in range(args.interpolation_scale)
        ]

        model_infer_average_interval: Interval = Interval()
        pipeline_fps = FPS()

        model = get_core(
            use_tensorrt=args.use_tensorrt,
            model_version=args.model_version,
            model_name=args.model_name,
            model_seperable=args.model_seperable,
            model_half=args.model_half,
            model_cache_size=args.max_gpu_cache_len,
            model_use_eyebrow=args.eyebrow,
            use_interpolation=args.use_interpolation,
            interpolation_scale=args.interpolation_scale,
            interpolation_half=args.interpolation_half,
            cacher_ram_size=args.max_ram_cache_len,
            use_sr=args.use_sr,
            sr_half=args.sr_half,
            sr_x4=args.sr_x4,
            sr_a4k=args.sr_a4k,
        )
        model.setImage(self.input_image)
        model_infer_average_interval.start()
        model.inference([np.zeros((1, 45), dtype=np.float32)])
        model_infer_average_interval.stop()
        self.last_model_interval.value = model_infer_average_interval.last()

        last_pose = np.zeros((45,), dtype=np.float32)
        last_text_motion_log_time = 0.0

        print("Model Inference Ready")
        while True:
            with pose_position_shm_guard.lock():
                np_pose = np_pose_shm.copy()
                np_position = np_position_shm.copy()

            text_motion = {
                "mouth_open": 0.0,
                "mouth_smile": 0.0,
                "body_x": 0.0,
                "body_y": 0.0,
                "body_scale": 0.0,
                "body_rotation": 0.0,
            }
            apply_text_overlay = not args.text_input
            if apply_text_overlay:
                text_motion = _read_text_motion()
                now = time.perf_counter()
                if now - last_text_motion_log_time > 2.0:
                    print("Text motion mouth_open:", round(text_motion["mouth_open"], 3))
                    last_text_motion_log_time = now

            input_poses = []
            increment = (np_pose - last_pose) / args.interpolation_scale
            for i in range(args.interpolation_scale):
                model_input = last_pose + increment * (i + 1)
                if apply_text_overlay:
                    _apply_text_motion_to_model_input(model_input, text_motion)
                input_poses.append(pose_simplify(model_input))

            last_pose = np_pose

            model_infer_average_interval.start()
            output_images = model.inference(input_poses)

            if args.max_ram_cache_len > 0:
                hits = model.cacher.hits
                miss = model.cacher.miss
                if args.use_sr:
                    hits += model.sr_cacher.hits
                    miss += model.sr_cacher.miss
                total = hits + miss
                self.cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            if args.use_tensorrt and args.max_gpu_cache_len > 0:
                hits = model.tha.cacher.hits
                miss = model.tha.cacher.miss
                total = hits + miss
                self.gpu_cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            output_images = self.post_process_ret(np_position, output_images, text_motion)

            self.average_model_interval.value = model_infer_average_interval.stop()
            self.last_model_interval.value = model_infer_average_interval.last()

            self.pipeline_fps_number.value = pipeline_fps()
            for i in range(args.interpolation_scale):
                with ret_batch_shm_guard[i].lock():
                    np_ret_shms[i][:, :, :] = output_images[i]

            self.finish_event.set()

    def post_process_ret(self, np_position: np.ndarray, output_images: np.ndarray, text_motion: dict) -> List[np.ndarray]:
        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = np_position[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -np_position[0] * 10 * args.extend_movement
            dx = np_position[0] * 400 * k_scale * args.extend_movement
            dy = -np_position[1] * 600 * k_scale * args.extend_movement
        k_scale *= 1.0
        rotate_angle += 0.0
        dx += 0.0
        dy += 0.0
        if args.bongo:
            rotate_angle -= 5

        rm = cv2.getRotationMatrix2D(
            (output_images[0].shape[1] / 2, output_images[0].shape[0] / 2),
            rotate_angle,
            k_scale,
        )
        rm[0, 2] += dx
        rm[1, 2] += dy

        ret = []
        for i in range(output_images.shape[0]):
            bgra_image = output_images[i]
            bgra_image = cv2.warpAffine(
                bgra_image,
                rm,
                (bgra_image.shape[1], bgra_image.shape[0]),
            )

            if args.output_debug:
                y = 16
                cv2.putText(
                    bgra_image,
                    "INFER/S: {:.4f}".format(self.pipeline_fps_number.value),
                    (0, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    1,
                )
                y += 16
                cv2.putText(
                    bgra_image,
                    "INPUT/S: {:.4f}".format(self.input_fps.value),
                    (0, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    1,
                )
                y += 16
                cv2.putText(
                    bgra_image,
                    "OUTPUT/S: {:.4f}".format(self.output_pipeline_fps.value),
                    (0, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    1,
                )
                y += 16
                cv2.putText(
                    bgra_image,
                    "CALC: {:.2f} ms".format(self.average_model_interval.value * 1000),
                    (0, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    1,
                )
                y += 16
                if args.max_ram_cache_len > 0:
                    cv2.putText(
                        bgra_image,
                        "MEM CACHE: {:.2f}%".format(self.cache_hit_ratio.value * 100),
                        (0, y),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 0),
                        1,
                    )
                    y += 16
                if args.max_gpu_cache_len > 0:
                    cv2.putText(
                        bgra_image,
                        "GPU CACHE: {:.2f}%".format(self.gpu_cache_hit_ratio.value * 100),
                        (0, y),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 0),
                        1,
                    )

            if args.alpha_split:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                alpha_channel = rgba_image[:, :, 3]
                rgb_channels = rgba_image[:, :, :3]
                alpha_image = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2RGB)
                rgb_channels = cv2.hconcat([rgb_channels, alpha_image])

            if args.output_debug:
                if args.alpha_split:
                    bgr_channels = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2BGR)
                else:
                    bgr_channels = _composite_bgra_over_white(bgra_image)
                ret.append(bgr_channels)
            elif args.output_virtual_cam:
                if not args.alpha_split:
                    rgb_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)
                ret.append(rgb_channels)
            else:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                ret.append(rgba_image)
        return ret


if __name__ == "__main__":
    pass
