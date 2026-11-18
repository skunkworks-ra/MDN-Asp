"""Convert a pickle .pt training checkpoint to a HF-safe safetensors model.

Training checkpoints (scripts/train_mdn_asp.py) are saved with torch.save as
Python pickle: {"step", "model": state_dict, ["val_loss"]}.  Pickle is flagged
"unsafe" by the Hugging Face scanner because it can execute arbitrary code on
load.  This script reconstructs the MDNAsp module from the state dict and
re-serialises it with PyTorchModelHubMixin.save_pretrained, which writes
model.safetensors + config.json (no pickle, no banner).

Usage
-----
    python scripts/convert_checkpoint_to_hf.py \
        --ckpt results/mdn_asp_v2/best.pt \
        --out_dir results/mdn_asp_v2/hf

Optionally push to the Hub (requires `huggingface-cli login` first):
        --push_to_hub --repo_id <user>/mad-clean-mdn-asp

The config args must match the architecture the checkpoint was trained with.
The defaults below match scripts/train_mdn_asp.py defaults.

--diffusion: a scripts/train_tiled_asp.py checkpoint instead. Writes
model.safetensors + config.json ({"step", "args"}); run_imaging.py
--diffusion_ckpt / --diff_ld_ckpt accept the output directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mdn_asp.models.mdn_asp import MDNAsp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert .pt checkpoint to HF safetensors.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the pickle .pt checkpoint.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for model.safetensors + config.json.")
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--hidden",        type=int, default=None)
    p.add_argument("--n_components",  type=int, default=None)
    p.add_argument("--cond_dim",      type=int, default=5)
    p.add_argument("--cutout",        type=int, default=None)
    p.add_argument("--pa_head",       choices=("7d", "von_mises"), default=None)
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push the converted model to the Hub (needs login).")
    p.add_argument("--repo_id", type=str, default=None,
                   help="Hub repo id, e.g. <user>/mad-clean-mdn-asp.")
    p.add_argument("--diffusion", action="store_true",
                   help="Diffusion checkpoint (train_tiled_asp.py, variants P / LD / A / B).")
    return p.parse_args(argv)


def convert_diffusion(args):
    """State dict -> model.safetensors, training args -> config.json. No push (subfolder upload with `hf upload`)."""
    import json
    from safetensors.torch import save_file
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)   # args are a plain dict
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck["model"].items()}, str(out_dir / "model.safetensors"))
    with open(out_dir / "config.json", "w") as f:
        json.dump({"step": ck.get("step"), "args": ck["args"]}, f, indent=2)
    src_rc = Path(args.ckpt).resolve().parent / "run_config.json"
    if src_rc.is_file():
        import shutil
        shutil.copy2(src_rc, out_dir / "run_config.json")
    print(f"[convert] Wrote model.safetensors + config.json to {out_dir}/ (step {ck.get('step')}, "
          f"variant {ck['args']['variant']})")


def main(argv=None):
    import json
    import shutil
    args = parse_args(argv)
    if args.diffusion:
        convert_diffusion(args)
        return

    # weights_only=True is safe here: we only need the tensor state dict and primitive metadata.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    train_args: dict = {}
    rc_data: dict | None = None
    src_rc = Path(args.ckpt).resolve().parent / "run_config.json"
    if src_rc.is_file():
        with open(src_rc) as fh:
            rc_data = json.load(fh)
        train_args = rc_data.get("args", {}) if isinstance(rc_data, dict) else {}
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        train_args = dict(ckpt["args"])
        rc_data = {"args": train_args}

    scale_relative = train_args.get("scale_relative")
    if scale_relative is None and isinstance(ckpt, dict) and "scale_relative" in ckpt:
        scale_relative = bool(ckpt["scale_relative"])
    point_sigma_px = train_args.get("point_sigma_px")
    pa_head = args.pa_head or train_args.get("pa_head", "7d")

    model = MDNAsp(
        base_channels=args.base_channels if args.base_channels is not None else train_args.get("base_channels", 32),
        hidden=args.hidden if args.hidden is not None else train_args.get("hidden", 256),
        n_components=args.n_components if args.n_components is not None else train_args.get("n_components", 5),
        cond_dim=args.cond_dim,
        cutout=args.cutout if args.cutout is not None else train_args.get("cutout", 128),
        pa_head=pa_head,
        scale_relative=scale_relative,
        point_sigma_px=point_sigma_px,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"State dict mismatch -- architecture args likely wrong.\n"
            f"  missing:    {missing}\n  unexpected: {unexpected}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    if src_rc.is_file():
        shutil.copy2(src_rc, out_dir / "run_config.json")
    elif rc_data is not None:
        with open(out_dir / "run_config.json", "w") as fh:
            json.dump(rc_data, fh, indent=2)
    print(f"[convert] Wrote safetensors model + config.json to {out_dir}/ "
          f"(scale_relative={scale_relative}, point_sigma_px={point_sigma_px}, pa_head={pa_head})")

    if args.push_to_hub:
        if not args.repo_id:
            raise SystemExit("--push_to_hub requires --repo_id")
        model.push_to_hub(args.repo_id)
        if (out_dir / "run_config.json").is_file():
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(out_dir / "run_config.json"),
                path_in_repo="run_config.json",
                repo_id=args.repo_id,
            )
        print(f"[convert] Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
