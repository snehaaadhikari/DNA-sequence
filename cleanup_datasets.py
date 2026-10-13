"""
Tidy the DataSets folder for the human-vs-chimp project.

Keeps ONLY the four source files. Everything else is MOVED (not deleted) into
DataSets/_old so you can restore anything by dragging it back.

  python cleanup_datasets.py "D:\\TECH405\\DNASequences\\DataSets"          # preview only
  python cleanup_datasets.py "D:\\TECH405\\DNASequences\\DataSets" --apply  # do it
"""
import argparse
import shutil
from pathlib import Path

KEEP = {"human_cds.fna", "human_clean.csv", "chimpanzee_cds.fna", "chimpanzee_clean.csv"}
OLD = "_old"

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--apply", action="store_true", help="actually move the files")
args = ap.parse_args()

root = Path(args.folder)
missing = [k for k in KEEP if not (root / k).exists()]
if missing:
    raise SystemExit(f"Stopping: these expected files are missing: {missing}")

old = root / OLD
moved = 0
for p in sorted(root.iterdir()):
    if p.name in KEEP or p.name == OLD:
        continue
    print(("MOVE  " if args.apply else "would move  ") + p.name)
    if args.apply:
        old.mkdir(exist_ok=True)
        shutil.move(str(p), str(old / p.name))
    moved += 1

print(f"\nKept: {sorted(KEEP)}")
print(f"{moved} item(s) {'moved to' if args.apply else 'would be moved to'} {OLD}/")
if not args.apply:
    print("Preview only. Re-run with --apply to do it.")
