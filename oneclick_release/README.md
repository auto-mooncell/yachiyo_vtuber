# One-Click Release Layer

This directory is a thin release layer on top of the existing:

- `EasyVtuber`
- `GPT-SoVITS-v2pro-20250604`

It does not duplicate the two base projects. Instead, it provides:

- fixed defaults
- a simplified input directory layout
- a review pause for `.list` correction
- a repeatable training and launch workflow
- a switchable local / remote GPT-SoVITS launch path

## Directory Layout

- `data/input/audio`: put one training audio file here
- `data/input/image`: put one character `.png` here
- `data/working/lists`: generated and corrected `.list` files
- `data/output`: saved session metadata
- `config/defaults.json`: release defaults

## First Workflow

1. Put one audio file into `data/input/audio`
2. Put one `.png` image into `data/input/image`
3. Run:

```powershell
cd "c:\Users\njh\Desktop\academic document\yachiyo_vtuber\oneclick_release"
python start.py full --character demo_char
```

This will:

- copy the image into `EasyVtuber/data/images`
- slice and ASR the audio
- create `data/working/lists/demo_char.list`
- open the `.list` in Notepad
- wait for you to correct and confirm
- train GPT-SoVITS
- choose the first valid list row as the reference audio/text
- write `data/output/current_session.json`

## Launch Workflow

After training finishes:

```powershell
cd "c:\Users\njh\Desktop\academic document\yachiyo_vtuber\oneclick_release"
python start.py launch --interactive
```

This will:

- start the GPT-SoVITS API in local mode, or reuse a configured remote API in remote mode
- switch API weights to the newly trained character
- start EasyVtuber with fixed text-input settings
- optionally start the interactive bridge in the current console

## Local And Remote Modes

`config/defaults.json` now supports two GPT-SoVITS modes:

- `local`: this machine launches `api_v2.py` and does TTS locally
- `remote`: this machine keeps running EasyVtuber locally, but sends TTS requests to an external GPT-SoVITS API

Relevant config keys:

```json
"gpt_sovits": {
  "mode": "local",
  "api_host": "127.0.0.1",
  "api_port": 9880,
  "remote_api_base": "https://example.com",
  "manage_remote_weights": true
}
```

Recommended usage:

- Keep `mode: local` if the user has a usable local GPU
- Set `mode: remote` if TTS should run on AutoDL / HPC / another GPU server
- Set `manage_remote_weights: false` if the remote server already loads the correct character itself and should not receive `/set_*_weights` calls

In remote mode, this release layer still keeps:

- EasyVtuber local
- image display local
- motion playback local
- optional LLM local

Only the GPT-SoVITS API is moved off-machine.

## Current Limits

- image input currently expects a `.png`
- the first valid row in the corrected `.list` is used as the default reference
- reference selection now requires existing audio, non-empty text, and a duration between 3 and 10 seconds
- this layer assumes the base repos stay in the current workspace layout
- training is still local for now; remote mode currently covers inference only
