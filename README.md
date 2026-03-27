# Yachiyo VTuber Minimal Repo

This repository is the smallest version of the project that still preserves the pipeline we already ran successfully:

- `oneclick_release` orchestration
- GPT-SoVITS training and TTS inference
- EasyVtuber text-input output and desktop-pet rendering

It is intentionally not the full upstream toolchain. Non-pipeline branches such as WebUI entrypoints, extra launcher scripts, demo/test files, and unrelated runtime/cache outputs are excluded.

## What This Repo Supports

The current minimal repo is built around these commands:

```powershell
cd "c:\Users\njh\Desktop\academic document\yachiyo_vtuber\oneclick_release"
python start.py full --character sakiko
python start.py launch --interactive
```

This means the repo is scoped to:

- one-click preparation
- `.list` review and correction
- local GPT-SoVITS training
- local GPT-SoVITS API launch
- remote GPT-SoVITS API mode for inference
- local EasyVtuber rendering
- interactive bridge with LLM + TTS + VTuber output

## Current Default Stack

The current `oneclick_release/config/defaults.json` uses:

- GPT-SoVITS version: `v2Pro`
- GPT-SoVITS mode: `local` by default, `remote` supported for inference
- EasyVtuber output: `desktop_pet`
- EasyVtuber model version: `v3`
- interpolation: enabled, scale `4`, half precision
- super-resolution: enabled, half precision

Important detail: the current default EasyVtuber launch path does **not** pass `--sr_x4`, `--use_tensorrt`, `--model_half`, or `--model_seperable`.

So the actual default model usage is:

- THA model: `tha3/standard/fp32`
- RIFE models: `rife_x4_fp16`, plus `rife_x2_fp16` and `rife_x3_fp16`
- SR model: `waifu2x/noise0_scale2x_fp16`

It does **not** require THA4 or Real-ESRGAN for the current default oneclick path.

## Minimal GPT-SoVITS Assets Required

For the current `v2Pro` training + inference path, the required GPT-SoVITS pretrained assets are:

- `GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base/*`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/*`
- `GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin`

If you want a bundled default character, keep:

- `GPT-SoVITS/GPT_weights_v2Pro/八千代-e15.ckpt`
- `GPT-SoVITS/SoVITS_weights_v2Pro/八千代_e8_s168.pth`

## Minimal EasyVtuber Assets Required

For the current default oneclick launch path, the required EasyVtuber model files are:

- `EasyVtuber/data/models/tha3/standard/fp32/*`
- `EasyVtuber/data/models/rife/rife_x2_fp16.onnx`
- `EasyVtuber/data/models/rife/rife_x3_fp16.onnx`
- `EasyVtuber/data/models/rife/rife_x4_fp16.onnx`
- `EasyVtuber/data/models/waifu2x/noise0_scale2x_fp16.onnx`

The current default path does not require:

- `EasyVtuber/data/models/tha4/*`
- `EasyVtuber/data/models/Real-ESRGAN/*`
- `EasyVtuber/data/models/waifu2x/noise0_scale2x_fp32.onnx`
- the fp16/fp32 variants not referenced above

## Repo Layout

- `oneclick_release/`
  Thin release layer and fixed workflow entrypoint.
- `GPT-SoVITS/`
  Training and TTS inference backend kept only to the extent needed by the oneclick pipeline.
- `EasyVtuber/`
  Local VTuber output path kept only for text input, bridge control, and desktop-pet rendering.

## Local And Remote Easy Modes

This minimal repo still supports two easy-mode inference paths:

- `local`
  `oneclick_release` launches `GPT-SoVITS/api_v2.py` locally.
- `remote`
  `oneclick_release` keeps EasyVtuber local, but sends TTS requests to a remote GPT-SoVITS API.

Training is still local in this minimal version.

## Notes

- This repo is the minimal publishable version, not the final full-featured repo.
- More assets, models, and optional paths can be added later.
- The goal of this version is simple: keep `oneclick + training + EasyVtuber output` working with the smallest maintainable repository boundary.
