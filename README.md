# Generative Drug Discovery AI — Rule-Driven Seq2Seq + SELFIES

An end-to-end, PC-runnable AI pipeline that reads a disease protein's amino-acid sequence and generates candidate drug molecules as SELFIES/SMILES, then puts each candidate through a 15-stage computational validation gauntlet (RDKit → ADMET → docking → MD → free energy).

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-cheminformatics-green)
![SELFIES](https://img.shields.io/badge/representation-SELFIES-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research%20demo-yellow)

## Table of Contents

- [Overview](#overview)
- [Why this project?](#why-this-project)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The learning rule is not hardcoded](#the-learning-rule-is-not-hardcoded)
- [customremedies.py — the generator](#customremediespy--the-generator)
- [tryoutDrugs.py — the validator](#tryoutdrugspy--the-validator)
- [Training data](#training-data)
- [Example output](#example-output)
- [Limitations & responsible use](#limitations--responsible-use)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

## Overview

`customremedies.py` trains a Seq2Seq Transformer that maps a target protein's amino-acid sequence to a SELFIES molecule string, using a plasticity rule read at runtime from `learningrules.txt`. Once trained, you can paste any disease protein sequence and the model will emit a candidate inhibitor, decoded back to SMILES and instantly sanity-checked by RDKit.

`tryoutDrugs.py` then takes that candidate and runs it through the full in-silico drug-discovery checklist — physico-chemistry, drug-likeness filters, toxicity alerts, solubility, ADMET, off-target prediction, protein structure prediction, binding-pocket detection, docking, MD, free-energy estimation, and a wet-lab validation checklist.

| | |
|---|---|
| **Input** | Amino-acid sequence of a disease target protein (≥10 AAs) |
| **Output** | SELFIES → SMILES + QED + LogP + toxicity + docking + MD workflow |
| **Model** | Seq2Seq Transformer (4 enc + 4 dec layers, d_model=256, 8 heads) |
| **Representation** | SELFIES (100% syntactically valid by construction) |
| **Hardware** | CPU (default) or CUDA (auto-detected) |
| **Optimizer** | Custom `RuleBasedOptimizer` — Δw is eval-ed from a text file |

## Why this project?

Most generative-chemistry demos either:

- produce chemically invalid SMILES (broken rings, bad valence), or
- hard-code a single optimizer (Adam / SGD) so you can't experiment with biologically-plausible plasticity rules.

This repo solves both:

- **SELFIES guarantees every generated string decodes to a valid molecule.**
- **The plasticity rule is a text file.** You can rewrite Δw without touching the model code — try Hebbian, reward-modulated STDP, BCM, gradient descent, or your own invention, all in one line.

On top of generation, we ship a complete validation story so candidates don't live in a vacuum — they're screened the way a real medicinal-chemistry team would screen them.

## Pipeline at a glance

```
                ┌──────────────────────────────────────────────────────────────┐
                │                     customremedies.py                        │
                │                                                              │
   Protein AA ─▶│  ┌─────────┐    ┌──────────────────────┐    ┌──────────┐   │
   sequence     │  │ AA Vocab│──▶ │  Transformer Encoder │──▶ │ Decoder  │   │
   (≥10 AAs)    │  └─────────┘    └──────────────────────┘    └────┬─────┘   │
                │                                                    ▼         │
                │                                          SELFIES tokens      │
                │                                                    │         │
                │                                                    ▼         │
                │                                            ┌──────────────┐  │
                │                                            │  SELFIES →   │  │
                │                                            │   SMILES     │  │
                │                                            └──────┬───────┘  │
                └───────────────────────────────────────────────────┼──────────┘
                                                                     ▼
                ┌──────────────────────────────────────────────────────────────┐
                │                       tryoutDrugs.py                         │
                │                                                              │
                │  1.  RDKit validity & canonicalization                       │
                │  2.  Physchem (QED, LogP, TPSA, MW, HBD/HBA, rings, Fsp3)    │
                │  3.  Synthetic accessibility (SAscore)                       │
                │  4.  Drug-likeness (Lipinski, Veber, Ghose, Lead, PAINS)     │
                │  5.  Toxicity alerts (hERG, Ames, structural alerts)         │
                │  6.  Solubility (ESOL LogS)                                  │
                │  7.  ADMET quick profile                                     │
                │  8.  Off-target prediction (SwissTargetPrediction API)       │
                │  9.  Protein 3D structure (ESMFold API)                      │
                │  10. Binding-pocket detection (fpocket / P2Rank)             │
                │  11. Molecular docking (AutoDock Vina workflow)              │
                │  12. Molecular dynamics equilibration (OpenMM)               │
                │  13. Free-energy estimation (MM-GBSA / FEP)                  │
                │  14. Selectivity analysis (BLAST homologs)                   │
                │  15. Experimental validation checklist (15-step wet-lab)     │
                └──────────────────────────────────────────────────────────────┘
```

## Repository structure

```
.
├── customremedies.py        # Generative AI: AA seq → SELFIES/SMILES
├── tryoutDrugs.py            # Full in-silico validation pipeline
├── learningrules.txt         # The plasticity rule (editable, hot-loaded)
├── remedy_workspace/
│   ├── data/                 # Auto-generated training pairs
│   └── checkpoints/          # drug_gpt.pth + training_log.json
└── README.md                 # You are here
```

`learningrules.txt` is expected to contain a line of the form:

```
Δw = -ctx['err']
```

The right-hand side is eval-ed in a sandboxed namespace that exposes `jnp` (aliased to `torch`) and a per-weight context dict `ctx` with keys:

| Key | Meaning |
|---|---|
| `mem` | the current weight value (`p.data`) |
| `post` | the post-synaptic gradient |
| `err` | the gradient (same as `post` for plain GD) |
| `reward` | a scalar reward signal broadcast to the weight shape |

Update convention: `W_new = W + LR * Δw`.

## Installation

The script auto-installs missing packages on first run. If you prefer to do it manually:

```bash
# Core generator deps
pip install torch numpy tqdm selfies rdkit-pypi

# Optional validator deps (install as you need each stage)
pip install requests openmm meeko vina transformers
# SAscore: copy sascorer.py + rfscores from RDKit's Contrib folder
```

Tested on:

- Python 3.10.6
- Windows 11 
- CPU-only and CUDA 11.8+

## Quick start

```bash
git clone https://github.com/<your-user>/generative-drug-discovery.git
cd generative-drug-discovery
echo "Δw = -ctx['err']" > learningrules.txt   # plain gradient descent
python customremedies.py
```

You'll see:

```
[rule]  loaded from learningrules.txt:
[rule]  Δw = -ctx['err']
[rule]  sanity eval OK
[device] Using CPU (will be slower)
[dataset] Generating 10000 real Protein->Inhibitor training pairs...
[preprocess] Proteins shape: (10000, 110) | SELFIES shape: (10000, 60)
...
========================================================================
 RULE-DRIVEN SEQ2SEQ DRUG DISCOVERY
========================================================================
  Δw = -ctx['err']
  Epochs=4  Batch=16  LR=0.01
========================================================================
```

When training finishes (or when you load the checkpoint), an interactive prompt opens:

```
Enter Protein Sequence (AAs): demo
  [demo] Using SARS-CoV-2 Main Protease (length 105)
AI is designing a molecule (SELFIES)...

--- AI Generated Molecule ---
Target Length : 105 AAs
Gen SELFIES   : [C][C][=C]...

[validation] Running RDKit chemical validation...
  Status       : CHEMICALLY VALID
  SMILES       : CC1=CC=C(...)
  QED Score    : 0.7234  (>0.5 is drug-like)
  LogP         : 2.41    (<5 is ideal)
  Num Rings    : 3
```

Now pass the SELFIES / SMILES to the validator:

```bash
python tryoutDrugs.py
```

You'll get a 15-section report ending with a final drug-likeness score out of 8.

## The learning rule is not hardcoded

This is the headline feature. Drop a different rule into `learningrules.txt` and re-train — no Python edits required.

### Examples you can try

| Rule file content | Behaviour |
|---|---|
| `Δw = -ctx['err']` | Plain gradient descent (default fallback) |
| `Δw = -ctx['err'] + 0.01 * ctx['reward'] * ctx['post']` | Reward-modulated Hebbian term on top of GD |
| `Δw = -ctx['err'] * (1 + torch.tanh(ctx['reward']))` | Reward-gated gradient |
| `Δw = -ctx['err'] + 0.02 * ctx['mem'] * ctx['post']` | Oja-style Hebbian decay |
| `Δw = torch.clamp(-ctx['err'], -0.5, 0.5)` | Sign-clipped gradient |

### Sandbox namespace

```
jnp (=torch), np (=torch), torch,
abs, sqrt, exp, log, sigmoid, tanh,
minimum, maximum, clip, clamp,
mean, sum, where, zeros_like, ones_like,
pow, square, norm, sign, relu, softplus,
ctx (dict: mem, post, err, reward)
```

Built-ins are stripped (`__builtins__ = {}`) for safety. The rule is sanity-probed at startup with a zero-tensor; if it raises, the code falls back to `-ctx['err']`.

## customremedies.py — the generator

### Architecture

```
Protein AAs  ──▶  Embedding(d_model=256)
                      │
                      ▼
               PositionalEncoding
                      │
                      ▼
          ┌───────────────────────┐
          │  Transformer Encoder  │   × 4 layers
          │   (8 heads, FF=1024)  │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Transformer Decoder  │   × 4 layers  (causal mask)
          │   (8 heads, FF=1024)  │
          └───────────┬───────────┘
                      │
                      ▼
                  Linear → SELFIES vocab logits
                      │
                      ▼
                argmax / greedy decode
                      │
                      ▼
                   SELFIES string
                      │
                      ▼
               selfies.decoder → SMILES
                      │
                      ▼
               RDKit parse + QED
```

### Key design choices

- **SELFIES, not SMILES.** Every token the decoder emits is part of a formal grammar that cannot produce an invalid molecule. No more "broken ring" failures mid-generation.
- **Cross-entropy with label smoothing (0.1)** + `ignore_index=0` for padding.
- **Causal mask + padding mask** on the decoder; the padding mask flows through to the memory so the protein encoder's `<pad>` positions don't pollute decoding.
- **Rule-based optimizer** wraps every `p.grad` with the `ctx` dict and applies your custom Δw. Gradient clipping (`max_norm=1.0`) and Δw clipping (±5.0) keep training stable.
- **Reward signal.** A running EMA of the loss (`L_bar`) tracks the recent mean; the per-step reward is `(L_bar - L_t) / |L_bar|` clamped to `[-1, 1]`. Good for reward-modulated rules.

### Hyperparameters (defaults)

| Name | Value | Notes |
|---|---|---|
| `DEFAULT_EPOCHS` | 4 | Demo-friendly; bump to 20+ for real runs |
| `DEFAULT_BATCH_SIZE` | 16 | Seq2Seq is memory-heavy |
| `DEFAULT_LR` | 1e-2 | Do not increase — Transformers diverge above 0.01 with this rule |
| `d_model` | 256 | |
| `nhead` | 8 | |
| `num_encoder_layers` | 4 | |
| `num_decoder_layers` | 4 | |
| `dim_feedforward` | 1024 | |
| `dropout` | 0.1 | |
| `SEED` | 42 | Reproducible |

## tryoutDrugs.py — the validator

A 15-stage report that turns a single SELFIES string into a defensible go/no-go decision.

| # | Stage | Method / Tool |
|---|---|---|
| 1 | RDKit validity | `Chem.MolFromSmiles` + sanitization |
| 2 | Physchem | QED, MolLogP, TPSA, MW, HBD/HBA, rotatable bonds, rings, Fsp3 |
| 3 | Synthetic accessibility | SAscore (RDKit Contrib) with heuristic fallback |
| 4 | Drug-likeness | Lipinski Ro5, Veber, Ghose, Lead-likeness, PAINS |
| 5 | Toxicity | 8 SMARTS structural alerts + hERG / Ames / carc / hepto heuristics |
| 6 | Solubility | ESOL-style LogS |
| 7 | ADMET quick | Oral absorption, BBB, CYP450, PPB, metabolic stability |
| 8 | Off-target | SwissTargetPrediction web API |
| 9 | Protein structure | ESMFold API (≤1024 AAs) |
| 10 | Binding pocket | fpocket CLI / P2Rank / DoGSiteScorer |
| 11 | Docking | AutoDock Vina (Meeko prep) |
| 12 | MD | OpenMM (ff14SB + GAFF + TIP3P) |
| 13 | Free energy | MM-GBSA / FEP+ / umbrella sampling |
| 14 | Selectivity | BLAST homologs + docking panel |
| 15 | Experimental checklist | 15-step wet-lab workflow (synthesis → efficacy) |

### Final verdict

An 8-point drug-likeness score:

```
QED > 0.5      ✓
LogP < 5       ✓
MW < 500       ✓
TPSA < 140     ✓
No tox alerts  ✓
hERG ≠ HIGH    ✓
LogS > -4      ✓
SAscore < 6    ✓
─────────────────
Score: 6/8   → proceed to docking / MD / experimental pipeline
```

## Training data

The generator bootstraps a real-world dataset from four clinically validated Protein → Inhibitor pairs:

| Target protein | Source drug(s) |
|---|---|
| SARS-CoV-2 Main Protease (Mpro) | Boceprevir-like analogue |
| HIV-1 Protease | Ritonavir fragment, indinavir-like |
| EGFR Tyrosine Kinase Domain | Gefitinib, erlotinib |
| Influenza A Neuraminidase | Zanamivir, oseltamivir carboxylate |

`download_dataset()` synthesizes 10,000 pairs by sampling a target, picking one of its drugs, encoding it to SELFIES, and randomly appending a valid SELFIES fragment (`[C]`, `[O]`, `[N]`, `[=C]`, `[Branch1][C][C]`) to simulate molecular analogues.

> **Note:** This is a demonstration-scale dataset. For research use, replace `REAL_TARGETS` with ChEMBL / BindingDB pulls (the code is structured to accept any `List[Tuple[protein_seq, smiles]]`).

## Example output

```
========================================================================
 AI-Generated Drug Candidate — Validation Pipeline
========================================================================
 Protein length          : 206 AAs (target 206)
 SELFIES length          : 178 chars
 Expected SMILES         : CC1=CC=C(C=C1)C(=O)N[C@@H1](C2CC=CC=C2C)C=O
========================================================================

[1] RDKit chemical validation
   Status              : CHEMICALLY VALID
   Canonical SMILES    : CC1=CC=C(C=C1)C(=O)N[C@@H1](c2ccccc2)C=O

[2] Physico-chemical profile
   QED score          : 0.7234   (>0.5 drug-like)
   LogP               : 2.41     (<5 ideal)
   TPSA               : 49.33 Å² (<140 good)
   MW                 : 239.27 Da (<500 Lipinski)
   HBD / HBA          : 2 / 2
   Rotatable bonds    : 4 (<10 Veber)
   Rings / Aromatic   : 2 / 2
   Fsp3               : 0.12

[3] Synthetic accessibility
   SAscore            : 2.41  (<6 synthesizable)

[4] Drug-likeness rule sets
   Lipinski Ro5        : PASS
   Veber                : PASS
   Ghose                : PASS
   Lead-likeness        : PASS
   PAINS alerts         : none

[5] Toxicity / safety profiling
   Structural alerts   : ['Aldehyde (reactive, irritant)']
   hERG channel risk   : LOW
   Ames mutagenicity   : no alert
   Carcinogenicity     : low
   Hepatotoxicity      : low

[6] Aqueous solubility estimate
   Estimated LogS      : -2.18 mol/L
   Solubility (mg/L)   : 1583.51
   Verdict              : highly soluble

...

========================================================================
 FINAL VERDICT
========================================================================
   Drug-likeness score : 7/8
   Recommend docking/MD/experimental pipeline if score ≥ 6.
========================================================================
```

## Limitations & responsible use

This repository is a research and educational demo. It is not a substitute for medicinal-chemistry expertise, regulatory review, or wet-lab validation.

- The bundled training set is tiny (4 targets, augmented to 10k pairs). Generated molecules should be treated as hypotheses, not drug candidates.
- Rule-based toxicity alerts have high false-positive/negative rates — they flag patterns, not measured toxicity.
- Docking, MD, and free-energy stages are scaffolds — they print the exact commands you should run; they do not produce validated poses on their own.
- Do not ingest, synthesize, or distribute any molecule produced by this pipeline without proper laboratory and regulatory oversight.

## Roadmap

- [ ] Replace `REAL_TARGETS` with a ChEMBL puller (`chembl_webresource_client`)
- [ ] Beam search + nucleus sampling in `generate()`
- [ ] Multi-property reinforcement fine-tuning (reward = QED − α·SA − β·Tox)
- [ ] Integration with `diffdock` for end-to-end differentiable docking
- [ ] Streamlit / Gradio web UI
- [ ] Pre-trained checkpoint release (>50k ChEMBL pairs)
- [ ] Active-learning loop: dock → label → retrain

## Citation

If this project is useful in your research, please cite:

```bibtex
@misc{generative_drug_discovery_ai,
  title  = {Generative Drug Discovery AI: Rule-Driven Seq2Seq + SELFIES},
  author = {Stijn},
  year   = {2025},
  url    = {https://github.com/flowvaultcontact-png/DRUG-SMILESai}
}
```

Please also cite the underlying tools: SELFIES (Krenn et al., 2020), RDKit, ESMFold (Lin et al., 2022), AutoDock Vina (Eberhardt et al., 2021), OpenMM.

## License

MIT — see [LICENSE](LICENSE). The bundled drug SMILES and protein sequences are publicly available facts from the literature and are not covered by this license.

---

<sub>Built for open science. Generate responsibly.</sub>
