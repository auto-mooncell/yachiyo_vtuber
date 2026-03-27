# EasyVtuber In This Repo

This is not the original upstream EasyVtuber distribution README.

In this repository, `EasyVtuber` is kept only as the local rendering/output layer for the current minimal pipeline:

- text input driven interaction
- GPT-SoVITS bridge playback
- desktop-pet style transparent output
- local motion/rendering driven by the bridge and text motion files

## What Is Actually Used

The current oneclick launch path is:

```powershell
cd "c:\Users\njh\Desktop\academic document\yachiyo_vtuber\oneclick_release"
python start.py launch --interactive
```

Internally this starts:

```powershell
python -m src.main --text_input --character <name> --use_interpolation --interpolation_half --interpolation_scale 4 --model_version v3 --use_sr --sr_half --output_desktop_pet
```

Important detail: the current launch path does **not** pass:

- `--use_tensorrt`
- `--model_half`
- `--model_seperable`
- `--sr_x4`

So the actual default model dependency is:

- THA: `data/models/tha3/standard/fp32/*`
- RIFE: `data/models/rife/rife_x4_fp16.onnx`
- RIFE helper models: `rife_x2_fp16.onnx`, `rife_x3_fp16.onnx`
- SR: `data/models/waifu2x/noise0_scale2x_fp16.onnx`

The current oneclick path does **not** require:

- THA4
- Real-ESRGAN
- webcam / iFacialMocap / OpenSeeFace input paths
- old launcher scripts
- packaging/test files

## Files In Scope

For this repo, the important EasyVtuber files are:

- `gptsovits_bridge.py`
- `character_prompts.json`
- `launcher2.py`
- `src/args.py`
- `src/main.py`
- `src/model_infer_client.py`
- `src/text_input_client.py`
- `src/desktop_pet_window.py`
- `src/ezvtb_rt_interface.py`
- `src/utils/*`
- `ezvtuber-rt/*`
- `assets/*`

## Why This README Is Minimal

This repo is being trimmed to match a single working pipeline:

- oneclick preparation/training
- GPT-SoVITS inference
- EasyVtuber output

So this README is intentionally scoped to the subset that the current repo actually uses, rather than documenting every upstream EasyVtuber feature branch.
