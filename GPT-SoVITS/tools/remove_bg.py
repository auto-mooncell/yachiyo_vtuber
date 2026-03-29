import argparse
import shutil
from pathlib import Path

from tools.uvr5.mdxnet import MDXNetDereverb


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Remove background/accompaniment before GPT-SoVITS training.")
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument("--output", required=True, help="Output vocal-only wav path.")
    parser.add_argument("--format", default="wav", choices=["wav", "flac", "mp3", "m4a"])
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocal_dir = output_path.parent / "_vocal_tmp"
    other_dir = output_path.parent / "_other_tmp"
    vocal_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    separator = MDXNetDereverb(15)
    separator._path_audio_(str(input_path), str(other_dir), str(vocal_dir), args.format, is_hp3=False)

    generated = vocal_dir / f"{input_path.name}_main_vocal.{args.format}"
    if not generated.exists():
        raise FileNotFoundError(f"Expected cleaned vocal file not found: {generated}")

    shutil.copyfile(generated, output_path)
    print(f"Background-removed audio written to: {output_path}")


if __name__ == "__main__":
    main()
