import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

S2_TEMPLATE_MAP = {
    "v2Pro": ROOT / "GPT_SoVITS" / "configs" / "s2v2Pro.json",
    "v2ProPlus": ROOT / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json",
    "v2": ROOT / "GPT_SoVITS" / "configs" / "s2.json",
}

S1_TEMPLATE = ROOT / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"

PRETRAINED_S2G = {
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}

PRETRAINED_S2D = {
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
}

PRETRAINED_S1 = {
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "v2Pro": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}

SV_MODEL = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

S2_WEIGHT_ROOT = {
    "v2": "SoVITS_weights_v2",
    "v2Pro": "SoVITS_weights_v2Pro",
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
}

S1_WEIGHT_ROOT = {
    "v2": "GPT_weights_v2",
    "v2Pro": "GPT_weights_v2Pro",
    "v2ProPlus": "GPT_weights_v2ProPlus",
}


def run_command(command, env=None):
    if env is None:
        env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [
        str(ROOT),
        str(ROOT / "tools"),
        str(ROOT / "GPT_SoVITS"),
    ]
    merged = os.pathsep.join(extra_paths)
    env["PYTHONPATH"] = merged if not current_pythonpath else merged + os.pathsep + current_pythonpath
    printable = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)
    print(f"\n$ {printable}\n")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def require_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def build_prepare_env(args, list_file: Path, wav_dir: str):
    env = os.environ.copy()
    env.update(
        {
            "inp_text": str(list_file),
            "inp_wav_dir": wav_dir,
            "exp_name": args.exp_name,
            "opt_dir": f"logs/{args.exp_name}",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": args.gpu,
            "is_half": str(args.is_half),
            "version": args.version,
            "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "pretrained_s2G": PRETRAINED_S2G[args.version],
            "s2config_path": f"GPT_SoVITS/configs/{S2_TEMPLATE_MAP[args.version].name}",
            "sv_path": SV_MODEL,
        }
    )
    return env


def run_slice_and_asr(args):
    output_slicer_dir = ROOT / "output" / "slicer_opt" / args.exp_name
    output_asr_dir = ROOT / "output" / "asr_opt"
    output_asr_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_existing_path(args.input_path)
    require_path(input_path, "Input path")

    if not args.skip_slice:
        run_command(
            [
                str(PYTHON),
                "tools/slice_audio.py",
                str(input_path),
                str(output_slicer_dir),
                str(args.slice_threshold),
                str(args.slice_min_length),
                str(args.slice_min_interval),
                str(args.slice_hop_size),
                str(args.slice_max_sil_kept),
                str(args.slice_max),
                str(args.slice_alpha),
                "0",
                "1",
            ]
        )
        asr_input_dir = output_slicer_dir
    else:
        if input_path.is_file():
            raise ValueError("--skip-slice requires --input-path to be a directory")
        asr_input_dir = input_path

    if args.skip_asr:
        raise ValueError("--skip-asr requires --list-file; ASR output list is otherwise missing")

    if args.language in {"zh", "yue"}:
        run_command(
            [
                str(PYTHON),
                "tools/asr/funasr_asr.py",
                "-i",
                str(asr_input_dir),
                "-o",
                str(output_asr_dir),
                "-l",
                args.language,
                "-p",
                args.asr_precision,
            ]
        )
    else:
        run_command(
            [
                str(PYTHON),
                "tools/asr/fasterwhisper_asr.py",
                "-i",
                str(asr_input_dir),
                "-o",
                str(output_asr_dir),
                "-l",
                args.language,
                "-p",
                args.asr_precision,
                "-s",
                args.asr_model_size,
            ]
        )

    list_path = output_asr_dir / f"{asr_input_dir.name}.list"
    require_path(list_path, "ASR list")
    return list_path, asr_input_dir


def merge_prepare_outputs(exp_dir: Path):
    text_part = exp_dir / "2-name2text-0.txt"
    semantic_part = exp_dir / "6-name2semantic-0.tsv"
    require_path(text_part, "1A output")
    require_path(semantic_part, "1C output")

    shutil.copyfile(text_part, exp_dir / "2-name2text.txt")
    semantic_lines = semantic_part.read_text(encoding="utf-8").strip().splitlines()
    semantic_output = exp_dir / "6-name2semantic.tsv"
    semantic_output.write_text(
        "item_name\tsemantic_audio\n" + "\n".join(semantic_lines) + "\n",
        encoding="utf-8",
    )


def create_s2_config(args, exp_dir: Path):
    template_path = S2_TEMPLATE_MAP[args.version]
    with template_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    (exp_dir / f"logs_s2_{args.version}").mkdir(parents=True, exist_ok=True)

    data["train"]["batch_size"] = args.s2_batch_size
    data["train"]["epochs"] = args.s2_epochs
    data["train"]["pretrained_s2G"] = PRETRAINED_S2G[args.version]
    data["train"]["pretrained_s2D"] = PRETRAINED_S2D[args.version]
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = args.s2_save_every_epoch
    data["train"]["gpu_numbers"] = args.gpu
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = args.lora_rank
    data["data"]["exp_dir"] = f"logs/{args.exp_name}"
    data["s2_ckpt_dir"] = f"logs/{args.exp_name}"
    data["save_weight_dir"] = S2_WEIGHT_ROOT[args.version]
    data["name"] = args.exp_name
    data["model"]["version"] = args.version
    data["version"] = args.version

    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    output_path = tmp_dir / f"{args.exp_name}_s2_{args.version}.json"
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def create_s1_config(args, exp_dir: Path):
    with S1_TEMPLATE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    (exp_dir / f"logs_s1_{args.version}").mkdir(parents=True, exist_ok=True)

    data["train"]["batch_size"] = args.s1_batch_size
    data["train"]["epochs"] = args.s1_epochs
    data["train"]["save_every_n_epoch"] = args.s1_save_every_n_epoch
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = S1_WEIGHT_ROOT[args.version]
    data["train"]["exp_name"] = args.exp_name
    data["pretrained_s1"] = PRETRAINED_S1[args.version]
    data["train_semantic_path"] = f"logs/{args.exp_name}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"logs/{args.exp_name}/2-name2text.txt"
    data["output_dir"] = f"logs/{args.exp_name}/logs_s1_{args.version}"

    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    output_path = tmp_dir / f"{args.exp_name}_s1_{args.version}.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="One-click GPT-SoVITS character training.")
    parser.add_argument("--exp-name", required=True, help="Character name. Used for logs and output weight filenames.")
    parser.add_argument("--input-path", help="Raw audio file or folder. Used for slicing/ASR when --list-file is not given.")
    parser.add_argument("--list-file", help="Existing training list. Skips slicing/ASR if provided.")
    parser.add_argument("--wav-dir", help="Directory for wav files referenced by --list-file when the list uses basenames.")
    parser.add_argument("--language", default="ja", help="ASR language. Example: ja, zh, en, yue.")
    parser.add_argument("--version", default="v2Pro", choices=["v2", "v2Pro", "v2ProPlus"])
    parser.add_argument("--gpu", default="0", help="GPU id string, for example 0.")
    parser.add_argument("--is-half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="is_half", action="store_false")
    parser.add_argument("--skip-slice", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--asr-model-size", default="large-v3")
    parser.add_argument("--asr-precision", default="float16")
    parser.add_argument("--slice-threshold", type=int, default=-40)
    parser.add_argument("--slice-min-length", type=int, default=5000)
    parser.add_argument("--slice-min-interval", type=int, default=300)
    parser.add_argument("--slice-hop-size", type=int, default=10)
    parser.add_argument("--slice-max-sil-kept", type=int, default=500)
    parser.add_argument("--slice-max", type=float, default=0.9)
    parser.add_argument("--slice-alpha", type=float, default=0.5)
    parser.add_argument("--s2-batch-size", type=int, default=4)
    parser.add_argument("--s2-epochs", type=int, default=8)
    parser.add_argument("--s2-save-every-epoch", type=int, default=4)
    parser.add_argument("--s1-batch-size", type=int, default=4)
    parser.add_argument("--s1-epochs", type=int, default=15)
    parser.add_argument("--s1-save-every-n-epoch", type=int, default=5)
    parser.add_argument("--lora-rank", default="32")
    args = parser.parse_args()

    if not args.list_file and not args.input_path:
        raise ValueError("Provide either --input-path or --list-file")

    exp_dir = ROOT / "logs" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.list_file:
        list_path = resolve_existing_path(args.list_file)
        require_path(list_path, "List file")
        wav_dir = ""
        if args.wav_dir:
            wav_dir_path = resolve_existing_path(args.wav_dir)
            require_path(wav_dir_path, "Wav directory")
            wav_dir = str(wav_dir_path)
        elif args.input_path:
            wav_dir_path = resolve_existing_path(args.input_path)
            if wav_dir_path.is_dir():
                wav_dir = str(wav_dir_path)
    else:
        if args.skip_asr:
            raise ValueError("--skip-asr requires --list-file")
        list_path, wav_dir_path = run_slice_and_asr(args)
        wav_dir = str(wav_dir_path)

    prepare_env = build_prepare_env(args, list_path, wav_dir)
    run_command([str(PYTHON), "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], env=prepare_env)
    run_command([str(PYTHON), "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env=prepare_env)
    if args.version in {"v2Pro", "v2ProPlus"}:
        run_command([str(PYTHON), "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], env=prepare_env)
    run_command([str(PYTHON), "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env=prepare_env)
    merge_prepare_outputs(exp_dir)

    s2_config = create_s2_config(args, exp_dir)
    run_command([str(PYTHON), "-s", "GPT_SoVITS/s2_train.py", "--config", str(s2_config)])

    s1_config = create_s1_config(args, exp_dir)
    run_command([str(PYTHON), "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(s1_config)])

    print("\nTraining completed.")
    print(f"Logs: {exp_dir}")
    print(f"SoVITS weights: {ROOT / S2_WEIGHT_ROOT[args.version]}")
    print(f"GPT weights: {ROOT / S1_WEIGHT_ROOT[args.version]}")


if __name__ == "__main__":
    main()
