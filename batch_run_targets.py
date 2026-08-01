"""
batch_run_targets.py
====================
Companion script for `customremedies.py`.

Reads a disease-targets catalog (e.g. `disease_targets.txt`) which contains
blocks of the form:

    >>>NAME: <short unique name>
    >>>SEQ: <one-line amino-acid sequence>

For every block, this script:
  1. Loads the trained checkpoint from remedy_workspace/checkpoints/drug_gpt.pth
     (only once, then reuses it for every target).
  2. Calls the model to generate a SELFIES candidate.
  3. Runs the FULL validation pipeline (RDKit, QED, drug-likeness, toxicity,
     solubility, ADMET, off-target prediction, protein structure, docking/MD
     scaffolding, selectivity, experimental checklist, aggregate verdict).
  4. Saves a per-target JSON report under remedy_workspace/reports/.
  5. Prints a one-line summary table at the end.

USAGE
-----
    python batch_run_targets.py --targets disease_targets.txt --load-only

    # Only run a subset (by name substring, case-insensitive):
    python batch_run_targets.py --targets disease_targets.txt --filter "Prion,ALS"

    # Skip slow network calls (ESMFold, SwissTargetPrediction):
    python batch_run_targets.py --targets disease_targets.txt --offline

    # Force retraining the model first:
    python batch_run_targets.py --targets disease_targets.txt --retrain

NOTE
----
This script imports from customremedies.py, so they MUST live in the same
directory. It does NOT modify the model — it just calls its public functions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure we can import customremedies.py living next to this script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import customremedies as cr  # noqa: E402

REPORT_DIR = cr.CKPT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Parser for the disease_targets.txt catalog
# ----------------------------------------------------------------------------
TARGET_BLOCK_RE = re.compile(
    r">>>NAME:\s*(?P<name>[^\n]+)\s*\n>>>SEQ:\s*(?P<seq>[ACDEFGHIKLMNPQRSTVWY]+)",
    re.IGNORECASE,
)


def parse_targets_file(path: Path) -> List[Tuple[str, str]]:
    """Return [(name, seq), ...] parsed from the catalog file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Only scan the region between >>>BEGIN TARGETS and >>>END TARGETS.
    # The MULTILINE flag makes ^ match at the start of each line, so the
    # markers must appear at the start of a line — this prevents accidental
    # matches inside quoted sentences elsewhere in the catalog file.
    m = re.search(
        r"^>>>BEGIN TARGETS\s*$.*?^>>>END TARGETS\s*$",
        text, re.DOTALL | re.MULTILINE,
    )
    body = m.group(0) if m else text
    targets = []
    for match in TARGET_BLOCK_RE.finditer(body):
        name = match.group("name").strip()
        seq = match.group("seq").strip().upper()
        targets.append((name, seq))
    if not targets:
        raise SystemExit(f"[batch] no >>>NAME:/>>>SEQ: blocks found in {path}")
    return targets


# ----------------------------------------------------------------------------
# Filter (for --filter flag)
# ----------------------------------------------------------------------------
def apply_filter(targets: List[Tuple[str, str]],
                 filter_str: Optional[str]) -> List[Tuple[str, str]]:
    if not filter_str:
        return targets
    needles = [n.strip().lower() for n in filter_str.split(",") if n.strip()]
    if not needles:
        return targets
    out = [(n, s) for (n, s) in targets
           if any(ndl in n.lower() for ndl in needles)]
    if not out:
        raise SystemExit(f"[batch] --filter matched nothing. Needles: {needles}")
    return out


# ----------------------------------------------------------------------------
# Load the model once, then reuse for every target
# ----------------------------------------------------------------------------
def load_model(device) -> Tuple[Any, Any, Any, Path]:
    ckpt_path = cr.CKPT_DIR / "drug_gpt.pth"
    if not ckpt_path.exists():
        raise SystemExit(
            f"[batch] checkpoint not found at {ckpt_path}. "
            f"Run customremedies.py once first to train & save the model."
        )
    ckpt = torch_load(ckpt_path, device)
    prot_vocab = _rebuild_vocab(ckpt["prot_vocab"])
    self_vocab = _rebuild_vocab(ckpt["self_vocab"])

    model = cr.DrugDiscoveryTransformer(
        prot_vocab_size=len(prot_vocab),
        self_vocab_size=len(self_vocab),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "rule" in ckpt:
        print(f"[batch] checkpoint trained with rule: Δw = {ckpt['rule']}")
    print(f"[batch] loaded model from {ckpt_path}")
    return model, prot_vocab, self_vocab, ckpt_path


def torch_load(ckpt_path: Path, device):
    import torch
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _rebuild_vocab(word2idx: Dict[str, int]):
    v = cr.Vocab.__new__(cr.Vocab)
    v.word2idx = dict(word2idx)
    v.idx2word = {i: w for w, i in word2idx.items()}
    return v


# ----------------------------------------------------------------------------
# Generate a SELFIES for one protein using the loaded model
# ----------------------------------------------------------------------------
def generate_selfies(model, prot_vocab, self_vocab, seq: str, device) -> str:
    import torch
    p_ids = prot_vocab.encode(seq) + [prot_vocab.word2idx["<eos>"]]
    src = torch.tensor([p_ids], dtype=torch.long, device=device)
    src_pad_mask = (src == 0)
    out_ids = model.generate(
        src, src_pad_mask,
        bos_idx=self_vocab.word2idx["<bos>"],
        eos_idx=self_vocab.word2idx["<eos>"],
        device=device,
    )[0].cpu().numpy().tolist()

    if out_ids and out_ids[0] == self_vocab.word2idx["<bos>"]:
        out_ids.pop(0)
    if self_vocab.word2idx["<eos>"] in out_ids:
        out_ids = out_ids[:out_ids.index(self_vocab.word2idx["<eos>"])]
    return "".join([self_vocab.idx2word.get(i, "") for i in out_ids])


# ----------------------------------------------------------------------------
# Run one target end-to-end
# ----------------------------------------------------------------------------
def run_one_target(name: str, seq: str, model, prot_vocab, self_vocab, device,
                   offline: bool) -> Dict[str, Any]:
    import torch
    from rdkit import Chem

    t0 = time.time()
    print("\n" + "#" * 72)
    print(f"# TARGET: {name}")
    print(f"# length={len(seq)} AA")
    print("#" * 72)

    report: Dict[str, Any] = {
        "name": name,
        "protein_length": len(seq),
        "selfies": None,
        "smiles": None,
        "valid": False,
        "qed": None,
        "logp": None,
        "druglikeness_score": None,
        "pipeline": None,
        "elapsed_sec": None,
    }

    try:
        gen_selfies = generate_selfies(model, prot_vocab, self_vocab, seq, device)
    except Exception as e:
        print(f"  [gen] ERROR: {e}")
        report["error"] = f"generation_failed: {e}"
        report["elapsed_sec"] = round(time.time() - t0, 2)
        return report

    report["selfies"] = gen_selfies
    print(f"  Gen SELFIES: {gen_selfies}")

    # Try to decode to SMILES up front so we can record it even if the
    # downstream pipeline fails for some reason.
    try:
        smiles = cr.sf.decoder(gen_selfies)
        report["smiles"] = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            report["valid"] = True
            report["qed"] = float(cr.QED.qed(mol))
            report["logp"] = float(cr.Descriptors.MolLogP(mol))
    except Exception as e:
        print(f"  [decode] WARNING: {e}")

    # Run the full validation pipeline (tryoutDrugs.py merged in).
    # When --offline is set, monkey-patch network calls to no-ops so the
    # batch doesn't hang on slow / blocked APIs.
    if offline:
        _patch_offline(cr)

    try:
        pipeline_result = cr.run_full_validation_pipeline(gen_selfies, seq)
        report["pipeline"] = _jsonify(pipeline_result)
        # Pull the headline drug-likeness score out for the summary table.
        if pipeline_result and "physchem" in pipeline_result:
            pc = pipeline_result["physchem"]
            # Recompute the 8-point score the same way aggregate_verdict does.
            score = 0
            if pc["qed"] > 0.5:               score += 1
            if pc["logp"] < 5:                score += 1
            if pc["mw"] < 500:                score += 1
            if pc["tpsa"] < 140:              score += 1
            if not pipeline_result.get("tox", {}).get("alerts"):
                score += 1
            if pipeline_result["tox"]["herg"] != "HIGH":
                score += 1
            if pipeline_result["sol"]["logS"] > -4:
                score += 1
            if pipeline_result["sa"] < 6:
                score += 1
            report["druglikeness_score"] = score
    except Exception as e:
        print(f"  [pipeline] ERROR: {e}")
        report["pipeline_error"] = str(e)

    report["elapsed_sec"] = round(time.time() - t0, 2)
    return report


def _patch_offline(cr_module) -> None:
    """Replace network-dependent functions with no-op stubs."""
    def _noop_off_target(smiles, *a, **kw):
        print("\n[8] Off-target prediction  (skipped: --offline)")
        return None

    def _noop_protein_structure(seq, out_pdb="protein.pdb", *a, **kw):
        print("\n[9] Protein structure  (skipped: --offline)")
        return None

    def _noop_binding_pocket(pdb_path, *a, **kw):
        print("\n[10] Binding pocket  (skipped: --offline)")
        return None

    cr_module.off_target_prediction = _noop_off_target
    cr_module.protein_structure = _noop_protein_structure
    cr_module.binding_pocket = _noop_binding_pocket


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain Python so the
    report can be JSON-serialized."""
    import numpy as np
    import torch
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
def print_summary(reports: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print(" BATCH SUMMARY")
    print("=" * 88)
    header = f"{'#':>3}  {'Name':<48}  {'Len':>4}  {'Valid':>5}  {'QED':>6}  {'Score':>5}"
    print(header)
    print("-" * 88)
    for i, r in enumerate(reports, 1):
        name = r.get("name", "?")[:48]
        L = r.get("protein_length", 0)
        valid = "yes" if r.get("valid") else "no"
        qed = f"{r['qed']:.3f}" if r.get("qed") is not None else "  -  "
        sc = str(r.get("druglikeness_score")) if r.get("druglikeness_score") is not None else "-"
        print(f"{i:>3}  {name:<48}  {L:>4}  {valid:>5}  {qed:>6}  {sc:>5}/8")
    print("=" * 88)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Batch-run customremedies.py over a catalog of disease targets.")
    ap.add_argument("--targets", type=str, default="disease_targets.txt",
                    help="Path to the targets catalog file.")
    ap.add_argument("--filter", type=str, default=None,
                    help="Comma-separated name substrings; only matching targets run.")
    ap.add_argument("--offline", action="store_true",
                    help="Skip ESMFold / SwissTargetPrediction / fpocket network calls.")
    ap.add_argument("--load-only", action="store_true",
                    help="Just load the existing checkpoint; do not prompt to retrain.")
    ap.add_argument("--retrain", action="store_true",
                    help="Force retraining before running the batch.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Where to write per-target JSON reports "
                         "(default: remedy_workspace/checkpoints/reports).")
    args = ap.parse_args()

    targets_path = Path(args.targets).expanduser().resolve()
    if not targets_path.exists():
        raise SystemExit(f"[batch] targets file not found: {targets_path}")
    targets = parse_targets_file(targets_path)
    targets = apply_filter(targets, args.filter)
    print(f"[batch] {len(targets)} target(s) selected from {targets_path.name}:")
    for n, s in targets:
        print(f"   - {n}  ({len(s)} AA)")

    # Determinism flags (mirror customremedies.main)
    import torch
    import numpy as np
    import random
    random.seed(cr.SEED)
    np.random.seed(cr.SEED)
    torch.manual_seed(cr.SEED)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    device = cr.pick_device()

    # Load (or train) the model exactly once.
    ckpt_path = cr.CKPT_DIR / "drug_gpt.pth"
    do_load = ckpt_path.exists() and not args.retrain
    if do_load:
        model, prot_vocab, self_vocab, _ = load_model(device)
    else:
        print("[batch] no checkpoint (or --retrain): training a fresh model...")
        items = cr.download_dataset()
        X, Y, prot_vocab, self_vocab = cr.build_dataset(items)
        model = cr.train_model(X, Y, prot_vocab, self_vocab, device,
                               epochs=cr.DEFAULT_EPOCHS,
                               batch_size=cr.DEFAULT_BATCH_SIZE,
                               lr=cr.DEFAULT_LR,
                               save_path=ckpt_path)

    # Run every target.
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for name, seq in targets:
        report = run_one_target(name, seq, model, prot_vocab, self_vocab,
                                device, offline=args.offline)
        reports.append(report)
        # Save per-target JSON immediately so partial progress isn't lost.
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        out_path = out_dir / f"{safe_name}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  [save] {out_path}")

    # Combined summary report.
    summary_path = out_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\n[batch] full summary written to {summary_path}")

    print_summary(reports)


if __name__ == "__main__":
    main()
