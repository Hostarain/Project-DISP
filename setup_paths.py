import re
from pathlib import Path

# ============================================================
DATASET_PATH = r"E:\data\dataset"
# ============================================================

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "mmdetection_github" / "configs"
SCRIPT_DIR = ROOT / "mmdetection_github"

BASE_PATH_ABS = "/home1/guojs/codes/Dark-ISP/mmdetection_github/configs/_base_/"
LOD_REL = "../../_base_/"
FLAT_REL = "../_base_/"

SCRIPTS = [
    SCRIPT_DIR / "LOD_RAW_preprocess.py",
    SCRIPT_DIR / "NOD_RAW_preprocess.py",
    SCRIPT_DIR / "COCO_Syn_preprocess.py",
    SCRIPT_DIR / "data_preparation" / "bayer_to_npy.py",
    SCRIPT_DIR / "test_dataset.py",
]


def replace_in_file(filepath: Path, pattern: str, replacement: str) -> None:
    content = filepath.read_text(encoding="utf-8")
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        filepath.write_text(new_content, encoding="utf-8")


def replace_fixed(filepath: Path, old: str, new: str) -> None:
    content = filepath.read_text(encoding="utf-8")
    new_content = content.replace(old, new)
    if new_content != content:
        filepath.write_text(new_content, encoding="utf-8")


def main():
    path = DATASET_PATH.replace("\\", "/").rstrip("/")

    print(f"=== Setting dataset root to: {path} ===")

    # 1. Update data_root in all config files
    print("[1/3] Updating data_root in configs...")
    pattern = r"data_root\s*=\s*'[^']*'"
    replacement = f"data_root = '{path}'"
    for py_file in CONFIG_DIR.rglob("*.py"):
        replace_in_file(py_file, pattern, replacement)
    print("       Done.")

    # 2. Fix _base_ paths from absolute to relative
    print("[2/3] Fixing _base_ paths...")
    subdirs = {
        "LOD": LOD_REL,
        "COCO": FLAT_REL,
        "NOD": FLAT_REL,
    }
    for subdir, rel_path in subdirs.items():
        target = CONFIG_DIR / subdir
        if target.is_dir():
            for py_file in target.rglob("*.py"):
                replace_fixed(py_file, BASE_PATH_ABS, rel_path)
    print("       Done.")

    # 3. Update dataset paths in scripts
    print("[3/3] Updating scripts...")
    for script in SCRIPTS:
        if script.is_file():
            content = script.read_text(encoding="utf-8")
            new_content = re.sub(
                r"'[^']*/(LOD|NOD|COCO|dataset)[^']*'",
                f"'{path}'",
                content,
            )
            new_content = re.sub(
                r"'/public/home/guojs/data/[^']*'",
                f"'{path}'",
                new_content,
            )
            if new_content != content:
                script.write_text(new_content, encoding="utf-8")
            print(f"       {script}")
    print("       Done.")

    print("=== Setup complete ===")


if __name__ == "__main__":
    main()
