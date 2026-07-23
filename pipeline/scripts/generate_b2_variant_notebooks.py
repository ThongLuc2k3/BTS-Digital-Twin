#!/usr/bin/env python3
import copy
import json
from pathlib import Path


ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"


DEPTH_PATCH_CELL = """# ==============================================
# CELL X - Patch train script for extra GS args
# ==============================================
from pathlib import Path

project_dir = Path(WORKDIR) / 'project'
train_script = project_dir / 'pipeline' / 'scripts' / '03_train_3dgs.sh'
text = train_script.read_text(encoding='utf-8')

if 'TRAIN_EXTRA_ARGS_RAW="${TRAIN_EXTRA_ARGS_RAW:-}"' not in text:
    text = text.replace(
        'CHECKPOINT_ITERATIONS_OVERRIDE="${CHECKPOINT_ITERATIONS_OVERRIDE:-}"\\n',
        'CHECKPOINT_ITERATIONS_OVERRIDE="${CHECKPOINT_ITERATIONS_OVERRIDE:-}"\\n'
        'TRAIN_EXTRA_ARGS_RAW="${TRAIN_EXTRA_ARGS_RAW:-}"\\n',
        1,
    )

if 'read -r -a TRAIN_EXTRA_ARGS' not in text:
    text = text.replace(
        'EXTRA_ARGS=()\\n',
        'TRAIN_EXTRA_ARGS=()\\n'
        'if [[ -n "$TRAIN_EXTRA_ARGS_RAW" ]]; then\\n'
        '  read -r -a TRAIN_EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS_RAW"\\n'
        'fi\\n\\n'
        'EXTRA_ARGS=()\\n',
        1,
    )

if '"${TRAIN_EXTRA_ARGS[@]}" \\\\' not in text:
    text = text.replace(
        '  --resolution "$RESOLUTION" \\\\\\n'
        '  "${EXTRA_ARGS[@]}" \\\\\\n',
        '  --resolution "$RESOLUTION" \\\\\\n'
        '  "${TRAIN_EXTRA_ARGS[@]}" \\\\\\n'
        '  "${EXTRA_ARGS[@]}" \\\\\\n',
        1,
    )

train_script.write_text(text, encoding='utf-8')
print('Patched train script for TRAIN_EXTRA_ARGS_RAW:', train_script)
"""


def variant_config_cell(enable_low_vram_override: bool, safe_densify_override: bool) -> str:
    lines = [
        "# =====================================================",
        "# CELL X - Configure depth supervision / train profile",
        "# =====================================================",
        "from pathlib import Path",
        "",
        "ENABLE_DEPTH_SUPERVISION = True",
        f"DISABLE_LOW_VRAM_PROFILE = {str(enable_low_vram_override)}",
        f"SAFE_DENSIFY_OVERRIDE = {str(safe_densify_override)}",
        "",
        "project_dir = Path(WORKDIR) / 'project'",
        "scene_root = project_dir / 'pipeline' / 'work' / SCENE",
        "depth_maps_dir = scene_root / 'colmap' / 'dense' / 'stereo' / 'depth_maps'",
        "",
        "extra_train_env = {}",
        "if gs_repo:",
        "    gs_repo_path = Path(gs_repo)",
        "    support_files = [",
        "        gs_repo_path / 'train.py',",
        "        gs_repo_path / 'arguments' / '__init__.py',",
        "        gs_repo_path / 'scene' / '__init__.py',",
        "    ]",
        "    support_text = '\\n'.join(",
        "        p.read_text(encoding='utf-8', errors='ignore')",
        "        for p in support_files",
        "        if p.exists()",
        "    )",
        "    if 'depths' not in support_text:",
        "        raise SystemExit(",
        "            'GS_REPO clone hien tai khong lo ro support --depths; notebook nay dung de bat depth supervision that nen se dung som.'",
        "        )",
        "    if ENABLE_DEPTH_SUPERVISION:",
        "        assert depth_maps_dir.exists(), f'Khong thay depth_maps: {depth_maps_dir}'",
        "        extra_train_env['TRAIN_EXTRA_ARGS_RAW'] = f'--depths {depth_maps_dir}'",
        "        print('Depth supervision args =', extra_train_env['TRAIN_EXTRA_ARGS_RAW'])",
        "    if DISABLE_LOW_VRAM_PROFILE:",
        "        extra_train_env['LOW_VRAM_PROFILE'] = '0'",
        "        print('LOW_VRAM_PROFILE override = 0')",
        "    elif SAFE_DENSIFY_OVERRIDE:",
        "        extra_train_env['LOW_VRAM_PROFILE'] = '1'",
        "        extra_train_env['DENSIFY_UNTIL_ITER'] = '15000'",
        "        extra_train_env['DENSIFY_FROM_ITER'] = '500'",
        "        extra_train_env['DENSIFICATION_INTERVAL'] = '100'",
        "        extra_train_env['PERCENT_DENSE'] = '0.01'",
        "        extra_train_env['OPACITY_RESET_INTERVAL'] = '3000'",
        "        print('SAFE densify overrides =', {",
        "            k: extra_train_env[k]",
        "            for k in [",
        "                'LOW_VRAM_PROFILE',",
        "                'DENSIFY_UNTIL_ITER',",
        "                'DENSIFY_FROM_ITER',",
        "                'DENSIFICATION_INTERVAL',",
        "                'PERCENT_DENSE',",
        "                'OPACITY_RESET_INTERVAL',",
        "            ]",
        "        })",
        "else:",
        "    print('gs_repo rong, se bo qua config depth/train cho den khi clone GS repo.')",
        "",
        "extra_train_env",
    ]
    return "\n".join(lines) + "\n"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Khong tim thay doan can thay: {old[:80]!r}")
    return text.replace(old, new, 1)


def to_source_block(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def build_variant(nb: dict, *, title_suffix: str, low_vram_off: bool, safe_densify: bool) -> dict:
    out = copy.deepcopy(nb)

    cell0 = "".join(out["cells"][0]["source"])
    out["cells"][0]["source"] = to_source_block(
        cell0.replace(
            "# BTS Digital Twin - B2 Pilot (`hcm0031`)",
            f"# BTS Digital Twin - {title_suffix} (`hcm0031`)",
        )
    )

    cell1 = "".join(out["cells"][1]["source"])
    cell1 = replace_once(cell1, "RUN_DENSE = '1'", "RUN_DENSE = '1'")
    cell1 = replace_once(cell1, "RUN_TRAIN = '0'", "RUN_TRAIN = '1'")
    cell1 = replace_once(cell1, "RUN_RENDER = '0'", "RUN_RENDER = '1'")
    cell1 = replace_once(cell1, "RUN_EVAL = '0'", "RUN_EVAL = '1'")
    out["cells"][1]["source"] = to_source_block(cell1)

    patch_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(DEPTH_PATCH_CELL),
    }
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(
            variant_config_cell(
                enable_low_vram_override=low_vram_off,
                safe_densify_override=safe_densify,
            )
        ),
    }

    out["cells"].insert(12, patch_cell)
    out["cells"].insert(13, config_cell)

    src = "".join(out["cells"][14]["source"])
    old = "if gs_repo:\n    env['GS_REPO'] = gs_repo\n"
    new = old + "env.update(extra_train_env)\n"
    out["cells"][14]["source"] = to_source_block(replace_once(src, old, new))

    src = "".join(out["cells"][22]["source"])
    old = "env['GS_REPO'] = str(GS_REPO)\n"
    new = old + "env.update(extra_train_env)\n"
    out["cells"][22]["source"] = to_source_block(replace_once(src, old, new))

    return out


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))

    variant1 = build_variant(
        nb,
        title_suffix="B2 Done 1 - Depth Supervision",
        low_vram_off=False,
        safe_densify=False,
    )
    variant2 = build_variant(
        nb,
        title_suffix="B2 Done 2 - Depth + No Low VRAM",
        low_vram_off=True,
        safe_densify=False,
    )
    variant3 = build_variant(
        nb,
        title_suffix="B2 Done 2 Safe - Depth + Safe Densify",
        low_vram_off=False,
        safe_densify=True,
    )

    out1 = ROOT / "downloads" / "B2_done 1.ipynb"
    out2 = ROOT / "downloads" / "B2_done 2.ipynb"
    out3 = ROOT / "downloads" / "B2_done 2 safe.ipynb"
    out1.write_text(json.dumps(variant1, ensure_ascii=False, indent=1), encoding="utf-8")
    out2.write_text(json.dumps(variant2, ensure_ascii=False, indent=1), encoding="utf-8")
    out3.write_text(json.dumps(variant3, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
