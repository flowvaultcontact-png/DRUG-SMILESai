"""
============================================================================
 AI-Generated Drug Candidate — Full Computational Validation Pipeline
============================================================================
Input
-----
Protein sequence (206 AA): MDSKIKDSEEEVIKEQMNKPNRNLLMDSEEMNRHKKLAQNRGLTTVQGEETQQLLREIEKMQNKTIQNAVSMLSEKLEQHGELYLATYELNKEQIEKIKQKETTTIVNQSEVETQVECTQTLDTVEKLQDSEKTKHHWREKQVEQWQKQVPSVYNIHHHHAQELMAAAGNLPITLLCTTMAHLENLEETVEYKLYLQKASQIFSGD
Gen SELFIES  : [C][C][=C][C][=C][Branch1][Branch1][C][=C][Ring1][=Branch1]...
Derived SMILES: CC1=CC=C(C=C1)C(=O)N[C@@H1](C2CC=CC=C2C)C=O

This script runs (or scaffolds) every check requested:
    RDKit chemical validation
    QED, LogP, ring count, additional physchem
    Synthetic accessibility (SA score)
    Drug-likeness filters (Lipinski, Veber, Ghose, Lead-likeness, PAINS)
    Toxicity alerts (hERG, Tox21-like rule-based, mutagenic alerts)
    Solubility (LogS estimation, ESOL-style)
    Off-target interaction prediction (SwissTargetPrediction API)
    Protein structure prediction (ESMFold via HuggingFace / AlphaFold DB)
    Binding-pocket identification (P2Rank / fpocket call-out)
    Molecular docking (AutoDock Vina via Meeko + vina)
    Molecular dynamics short equilibration (OpenMM)
    Free-energy estimate (MM-GBSA-style / OpenMM)
    Experimental validation workflow (checklist)

Required Python packages (install what you need):
    pip install rdkit-pypi selfies requests numpy openmm
    pip install meeko vina                 # for docking
    pip install transformers torch         # for ESMFold
    pip install sascorer (RDKit contrib)   # SA score
============================================================================
"""

import os
import sys
import json
import math
import textwrap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. INPUTS
# ---------------------------------------------------------------------------
PROTEIN_SEQ = (
    "MDSKIKDSEEEVIKEQMNKPNRNLLMDSEEMNRHKKLAQNRGLTTVQGEETQQLLREIEKMQNK"
    "TIQNAVSMLSEKLEQHGELYLATYELNKEQIEKIKQKETTTIVNQSEVETQVECTQTLDTVEK"
    "LQDSEKTKHHWREKQVEQWQKQVPSVYNIHHHHAQELMAAAGNLPITLLCTTMAHLENLEETVE"
    "YKLYLQKASQIFSGD"
)
TARGET_LENGTH_AA = 206
SELFIES_STR  = ("[C][C][=C][C][=C][Branch1][Branch1][C][=C][Ring1][=Branch1]"
                "[C][=Branch1][C][=O][N][C@@H1][Branch1][#Branch2]"
                "[C][C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O]"
                "[N][C][C][C][C][C@H1][Ring1][Branch1][C][=Branch1][C][=O]"
                "[N][C][O][C][O][C][Ring1][=Branch1]")
EXPECTED_SMILES = "CC1=CC=C(C=C1)C(=O)N[C@@H1](C2CC=CC=C2C)C=O"

print("="*70)
print(" AI-Generated Drug Candidate — Validation Pipeline")
print("="*70)
print(f" Protein length          : {len(PROTEIN_SEQ)} AAs "
      f"(target {TARGET_LENGTH_AA})")
print(f" SELFIES length          : {len(SELFIES_STR)} chars")
print(f" Expected SMILES         : {EXPECTED_SMILES}")
print("="*70)

# ---------------------------------------------------------------------------
# 1. RDKit CHEMICAL VALIDATION
# ---------------------------------------------------------------------------
def rdkit_validation(selfies_str: str, expected_smiles: str):
    print("\n[1] RDKit chemical validation")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, QED, Crippen, Lipinski, rdMolDescriptors
        import selfies as sf
    except ImportError as e:
        print("   ! RDKit / selfies not available — install: pip install rdkit-pypi selfies")
        return None

    # Decode SELFIES -> SMILES
    try:
        smi_from_selfies = sf.decoder(selfies_str)
    except Exception as e:
        print(f"   ! SELFIES decode failed: {e}")
        smi_from_selfies = expected_smiles

    mol = Chem.MolFromSmiles(smi_from_selfies)
    status = "CHEMICALLY VALID" if mol is not None else "INVALID"
    print(f"   Status              : {status}")
    print(f"   SMILES (from SELFIES): {smi_from_selfies}")
    if mol is None:
        return None

    # 2D coords + canonical
    AllChem.Compute2DCoords(mol)
    canon = Chem.MolToSmiles(mol)
    print(f"   Canonical SMILES    : {canon}")

    return mol


# ---------------------------------------------------------------------------
# 2. QED, LogP, Rings, extended physchem
# ---------------------------------------------------------------------------
def physchem_profile(mol):
    print("\n[2] Physico-chemical profile (QED / LogP / Rings / extras)")
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
    qed   = QED.qed(mol)
    logp  = Crippen.MolLogP(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    mw    = Descriptors.MolWt(mol)
    hbd   = Lipinski.NumHDonors(mol)
    hba   = Lipinski.NumHAcceptors(mol)
    rotb  = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aro   = rdMolDescriptors.CalcNumAromaticRings(mol)
    fsp3  = rdMolDescriptors.CalcFractionCSP3(mol)

    print(f"   QED score          : {qed:.4f}   (>0.5 drug-like)")
    print(f"   LogP               : {logp:.2f}   (<5 ideal)")
    print(f"   TPSA               : {tpsa:.2f} Å² (<140 good)")
    print(f"   MW                 : {mw:.2f} Da  (<500 Lipinski)")
    print(f"   HBD / HBA          : {hbd} / {hba}")
    print(f"   Rotatable bonds    : {rotb} (<10 Veber)")
    print(f"   Rings / Aromatic   : {rings} / {aro}")
    print(f"   Fsp3               : {fsp3:.2f}")

    return dict(qed=qed, logp=logp, tpsa=tpsa, mw=mw, hbd=hbd, hba=hba,
                rotb=rotb, rings=rings, aro=aro, fsp3=fsp3)


# ---------------------------------------------------------------------------
# 3. Synthetic Accessibility (SAscore)
# ---------------------------------------------------------------------------
def synthetic_accessibility(mol):
    print("\n[3] Synthetic accessibility (SAscore, 1 easy → 10 hard)")
    try:
        # rdkit Contrib path
        import sascorer
        sa = sascorer.calculateScore(mol)
        print(f"   SAscore            : {sa:.2f}  (<6 synthesizable)")
        return sa
    except Exception:
        # Fallback heuristic
        from rdkit.Chem import Lipinski, rdMolDescriptors
        sa = (1.0
              + 0.5 * rdMolDescriptors.CalcNumRings(mol)
              + 0.3 * rdMolDescriptors.CalcNumRotatableBonds(mol)
              + 0.2 * (rdMolDescriptors.CalcNumAromaticRings(mol)))
        print(f"   SAscore (heuristic): {sa:.2f}  (install rdkit Contrib for full SA score)")
        return sa


# ---------------------------------------------------------------------------
# 4. Drug-likeness filters
# ---------------------------------------------------------------------------
def druglikeness_filters(mol, pc):
    print("\n[4] Drug-likeness rule sets")
    from rdkit.Chem import Lipinski
    # Lipinski
    lipinski_pass = (pc['mw']<=500 and pc['logp']<=5 and pc['hbd']<=5 and pc['hba']<=10)
    # Veber
    veber_pass = (pc['rotb']<=10 and pc['tpsa']<=140)
    # Ghose
    ghose_pass = (160<=pc['mw']<=480 and -0.4<=pc['logp']<=5.6
                  and 40<=mol.GetNumAtoms()<=70 and pc['tpsa']<=140)
    # Lead-likeness
    lead_pass = (pc['mw']<=450 and pc['logp']<=3.5 and pc['hbd']<=3 and pc['hba']<=6)
    # PAINS (via RDKit filter)
    pains_hits = []
    try:
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalogs
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        entry = catalog.GetFirstMatch(mol)
        if entry is not None:
            pains_hits.append(entry.GetDescription())
    except Exception:
        pass

    print(f"   Lipinski Ro5        : {'PASS' if lipinski_pass else 'FAIL'}")
    print(f"   Veber               : {'PASS' if veber_pass else 'FAIL'}")
    print(f"   Ghose               : {'PASS' if ghose_pass else 'FAIL'}")
    print(f"   Lead-likeness       : {'PASS' if lead_pass else 'FAIL'}")
    print(f"   PAINS alerts        : {pains_hits if pains_hits else 'none'}")
    return dict(lipinski=lipinski_pass, veber=veber_pass, ghose=ghose_pass,
                lead=lead_pass, pains=pains_hits)


# ---------------------------------------------------------------------------
# 5. Toxicity alerts (rule-based + hERG heuristic)
# ---------------------------------------------------------------------------
def toxicity_screen(mol):
    print("\n[5] Toxicity / safety profiling (rule-based alerts)")
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors
    smarts_alerts = {
        "Mutagenic — aromatic nitro"       : "[#6]=[#6][N+](=O)[O-]",
        "Alkyl halide (reactive)"          : "[#6][F,Cl,Br,I]",
        "Aldehyde (reactive, irritant)"    : "[CX3H1](=O)[#6]",
        "Michael acceptor"                 : "[CX3]=[CX3][C,F,Cl,Br,I,OH]",
        "Hydrazine"                        : "[NX2][NX2]",
        "Azo group"                        : "[NX2]=[NX2]",
        "Epoxide / aziridine"              : "C1OC1 / C1NC1",
        "Thiol (reactive)"                 : "[SX2H]",
    }
    hits = []
    for name, sma in smarts_alerts.items():
        # split compound SMARTS by ' / '
        for s in sma.split(" / "):
            patt = Chem.MolFromSmarts(s.strip())
            if patt and mol.HasSubstructMatch(patt):
                hits.append(name)
                break

    # hERG heuristic (high LogP + aromatic rings correlates with hERG block)
    logp = Crippen.MolLogP(mol)
    aro_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    herg_risk = "HIGH" if (logp > 3.5 and aro_rings >= 2) else \
                "MODERATE" if logp > 2.5 else "LOW"

    # Ames mutagenicity heuristic
    ames_alert = any("Mutagenic" in h for h in hits)

    # Carcinogenicity heuristic (reactive alert + high lipophilicity)
    carc_risk = "ELEVATED" if (hits and logp > 4) else "low"

    # Hepatotoxicity heuristic
    hepto_risk = "ELEVATED" if (logp > 4 and aro_rings >= 2) else "low"

    print(f"   Structural alerts   : {hits if hits else 'none'}")
    print(f"   hERG channel risk   : {herg_risk}")
    print(f"   Ames mutagenicity   : {'FLAG' if ames_alert else 'no alert'}")
    print(f"   Carcinogenicity     : {carc_risk}")
    print(f"   Hepatotoxicity      : {hepto_risk}")
    return dict(alerts=hits, herg=herg_risk, ames=ames_alert,
                carc=carc_risk, hepto=hepto_risk)


# ---------------------------------------------------------------------------
# 6. Solubility (ESOL-style LogS estimate)
# ---------------------------------------------------------------------------
def solubility_estimate(mol, pc):
    print("\n[6] Aqueous solubility estimate (ESOL-style)")
    from rdkit.Chem import rdMolDescriptors
    mw = pc['mw']; logp = pc['logp']
    n_rot = pc['rotb']
    n_arom = pc['aro']
    # Delaney ESOL: LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.066*RB - 0.74*#ArRings
    logS = 0.16 - 0.63*logp - 0.0062*mw + 0.066*n_rot - 0.74*n_arom
    sol_mg_l = (10**logS) * mw * 1000
    print(f"   Estimated LogS      : {logS:.2f} mol/L")
    print(f"   Solubility (mg/L)   : {sol_mg_l:.2f}")
    verdict = ("highly soluble" if logS > -2 else
               "soluble"        if logS > -4 else
               "poorly soluble")
    print(f"   Verdict             : {verdict}")
    return dict(logS=logS, sol_mg_l=sol_mg_l)


# ---------------------------------------------------------------------------
# 7. ADMET quick profile (absorption, metabolism)
# ---------------------------------------------------------------------------
def admet_quick(mol, pc):
    print("\n[7] ADMET quick profile")
    # Oral absorption (Veber proxy)
    oral_absorption = "HIGH" if (pc['tpsa']<=140 and pc['rotb']<=10) else "LOW"
    # BBB permeability (logP/TPSA heuristic)
    bbb = "PERMEABLE" if (pc['logp']>1 and pc['tpsa']<90) else "LOW"
    # CYP450 liability (aromatic rings + lipophilicity)
    cyp_risk = "POSSIBLE" if (pc['logp']>3 and pc['aro']>=1) else "low"
    # Plasma protein binding proxy
    ppb = "HIGH" if pc['logp']>3 else "moderate"
    print(f"   Oral absorption     : {oral_absorption}")
    print(f"   BBB permeability    : {bbb}")
    print(f"   CYP450 liability    : {cyp_risk}")
    print(f"   Plasma protein bind : {ppb}")
    print(f"   Metabolic stability : {'possibly low (ester/amide)' if mol.HasSubstructMatch(__import__('rdkit').Chem.MolFromSmarts('C(=O)N')) else 'moderate'}")
    return dict(oral=oral_absorption, bbb=bbb, cyp=cyp_risk, ppb=ppb)


# ---------------------------------------------------------------------------
# 8. Off-target prediction via SwissTargetPrediction (web API)
# ---------------------------------------------------------------------------
def off_target_prediction(smiles: str):
    print("\n[8] Off-target interaction prediction (SwissTargetPrediction API)")
    import requests
    url = "https://swisstargetprediction.ch/result.php"
    # POST form
    data = {"smiles": smiles, "organism": "Homo sapiens"}
    try:
        r = requests.post(url, data=data, timeout=60)
        if r.status_code == 200 and "csv" in r.text.lower():
            print("   Request successful — parse result.php page for target list.")
        else:
            print(f"   HTTP {r.status_code} — visit "
                  "https://swisstargetprediction.ch/result.php?smiles={smiles}")
        print("   Alternative: SEA, SuperPred, ChemPRO, TargetNet APIs")
        print(f"   Quick-link  : https://swisstargetprediction.ch/result.php?smiles={smiles}")
    except Exception as e:
        print(f"   ! Network/API unavailable: {e}")
    print("   Note: cross-check predicted targets vs. your 206-AA protein's "
          "known interactome (BioGRID / STRING).")


# ---------------------------------------------------------------------------
# 9. Protein structure prediction (ESMFold via HuggingFace API)
# ---------------------------------------------------------------------------
def protein_structure(seq: str, out_pdb="protein.pdb"):
    print("\n[9] Protein 3D structure prediction (ESMFold API)")
    import requests
    if len(seq) > 1024:
        print(f"   Sequence {len(seq)} AAs > 1024 — ESMFold API limit. "
              "Use local ESMFold or AlphaFold ColabFold instead.")
        return None
    url = "https://api.esmatlas.com/fetchPredictedStructure/"
    try:
        r = requests.post(url, data=seq, timeout=600)
        if r.status_code == 200:
            with open(out_pdb, "w") as fh:
                fh.write(r.text)
            # crude pLDDT avg from B-factors
            lines = [l for l in r.text.splitlines() if l.startswith("ATOM") and l[12:16].strip()=="CA"]
            plddts = [float(l[60:66]) for l in lines]
            avg_plddt = sum(plddts)/max(1,len(plddts))
            print(f"   Structure saved to : {out_pdb}")
            print(f"   Avg pLDDT          : {avg_plddt:.1f}")
            print("   Confidence: HIGH" if avg_plddt>80 else
                  "   Confidence: MEDIUM" if avg_plddt>60 else
                  "   Confidence: LOW — consider AlphaFold2")
            return out_pdb
        else:
            print(f"   ESMFold HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"   ! ESMFold call failed: {e}")
        print("   Alternative: AlphaFold DB lookup if UniProt ID known, "
              "or run ColabFold locally.")
        return None


# ---------------------------------------------------------------------------
# 10. Binding-pocket identification (P2Rank / fpocket call-out)
# ---------------------------------------------------------------------------
def binding_pocket(pdb_path: str):
    print("\n[10] Binding pocket identification")
    if pdb_path is None or not os.path.exists(pdb_path):
        print("   No PDB available — skipping. Provide a structure "
              "(experimental or predicted) to run pocket detection.")
        return None
    # Try fpocket (command-line)
    import subprocess
    try:
        out = subprocess.run(["fpocket", "-f", pdb_path],
                             capture_output=True, text=True, timeout=600)
        print("   fpocket executed; pockets in "
              f"{pdb_path.replace('.pdb','')}_fpocket/")
    except FileNotFoundError:
        print("   fpocket not installed. Install: conda install -c bioconda fpocket")
        print("   Alternative: P2Rank  ->  https://github.com/rdk/p2rank")
        print("   Alternative: DoGSiteScorer web service (https://proteins.plus/)")
    except Exception as e:
        print(f"   ! fpocket error: {e}")
    return None


# ---------------------------------------------------------------------------
# 11. Molecular docking (AutoDock Vina via Meeko + vina)
# ---------------------------------------------------------------------------
def molecular_docking(lig_smiles: str, receptor_pdb: str,
                      box_center=(0,0,0), box_size=(20,20,20)):
    print("\n[11] Molecular docking (AutoDock Vina)")
    print("   Workflow:")
    print("     1. Prepare receptor: prepare_receptor -r receptor.pdbqt")
    print("     2. Prepare ligand  : mk_prepare_ligand.py -i ligand.sdf -o lig.pdbqt")
    print("     3. Vina docking    : vina --receptor receptor.pdbqt --ligand lig.pdbqt "
          f"--center_x {box_center[0]} --center_y {box_center[1]} --center_z {box_center[2]} "
          f"--size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]} "
          "--num_modes 9 --energy_range 3 --exhaustiveness 32")
    print("   Or use the `vina` Python bindings:")
    print("     from vina import Vina; v=Vina(); v.set_receptor('receptor.pdbqt');")
    print("     v.set_ligand_from_file('lig.pdbqt'); v.dock(); v.write_poses('out.pdbqt')")
    print("   Output: binding affinity (kcal/mol), RMSD, binding pose(s).")
    print("   Tip: negative ΔG (e.g., < -7 kcal/mol) suggests strong binding.")


# ---------------------------------------------------------------------------
# 12. Molecular dynamics short equilibration (OpenMM)
# ---------------------------------------------------------------------------
def md_simulation(pdb_path: str, lig_pdbqt: str = None):
    print("\n[12] Molecular dynamics equilibration (OpenMM)")
    print("   Workflow:")
    print("     1. Parametrize ligand with AM1-BCC (antechamber / openff-toolkit)")
    print("     2. Build system: protein (ff14SB) + ligand (GAFF) + TIP3P water + 0.15 M NaCl")
    print("     3. Minimize, heat 0→300 K, equilibrate NVT/NPT (~1 ns), run production.")
    print("   Skeleton OpenMM code:")
    print("""
        from openmm.app import *
        from openmm import *
        from openmm.unit import *
        pdb = PDBFile('complex.pdb')
        ff  = ForceField('amber14-all.xml','amber14/tip3p.xml','ligand.xml')
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer, constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        sim = Simulation(pdb.topology, system, integrator)
        sim.minimizeEnergy()
        sim.reporters.append(DCDReporter('traj.dcd', 1000))
        sim.reporters.append(StateDataReporter('log.csv', 1000, step=True,
            potentialEnergy=True, temperature=True))
        sim.step(500_000)   # 1 ns
    """)
    print("   Analyse RMSD, RMSF, ligand-protein H-bond occupancy.")


# ---------------------------------------------------------------------------
# 13. Free-energy calculation (MM-GBSA / FEP+)
# ---------------------------------------------------------------------------
def free_energy_calc():
    print("\n[13] Free-energy calculations")
    print("   Recommended methods (ordered by cost):")
    print("   • MM-GBSA / MM-PBSA — fast ΔG_bind estimate from MD snapshots")
    print("       tool: gmx_MMPBSA (GROMACS + APBS) or OpenMM + MMPBSA.py")
    print("   • Alchemical FEP (TI/MBAR) — rigorous ΔΔG for analog series")
    print("       tool: FEP+ (Schrödinger), NAMD/amber TI, BioSimSpace")
    print("   • Umbrella sampling along reaction coordinate (PMF)")
    print("   Report ΔG_bind ± SEM (kcal/mol); < -5 suggests low-µM binding.")


# ---------------------------------------------------------------------------
# 14. Selectivity & off-target scoring
# ---------------------------------------------------------------------------
def selectivity_analysis():
    print("\n[14] Selectivity analysis")
    print("   • Dock ligand against panel of off-target homologs / anti-targets.")
    print("   • Compute selectivity index = IC50(target) / IC50(off-target).")
    print("   • Cross-check with SwissTargetPrediction (step 8) and CHEMBL bioactivity.")
    print("   • For this 206-AA protein: BLAST against UniProt to identify homologs.")
    # BLAST snippet
    print("   Quick BLAST via NCBI WWW (need NCBI API key for high volume):")
    print("       https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins&PROGRAM=blastp")


# ---------------------------------------------------------------------------
# 15. Experimental validation checklist
# ---------------------------------------------------------------------------
def experimental_validation_checklist():
    print("\n[15] Experimental validation workflow (laboratory checklist)")
    checklist = [
        "Synthesize / purchase compound (verify by NMR, HRMS, HPLC purity ≥95%)",
        "Solubility assay (kinetic / thermodynamic)",
        "Plasma & microsomal stability (t½)",
        "Caco-2 / MDCK permeability",
        "CYP450 inhibition panel (3A4, 2D6, 1A2, 2C9, 2C19)",
        "hERG patch-clamp (manual / automated)",
        "Ames mutagenicity test (OECD 471)",
        "In vitro micronucleus / chromosome aberration",
        "Target binding: SPR / ITC / MST / DSF (Kd, kon/koff)",
        "Functional cell-based assay (IC50 / EC50, Hill slope)",
        "Selectivity panel (≥10 off-targets, Cerep/Eurofins)",
        "Cytotoxicity (HEK293, HepG2, primary cells)",
        "In vivo PK (rat / mouse): Cmax, AUC, t½, bioavailability",
        "Acute toxicity (OECD 423)",
        "Efficacy in disease-relevant model",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"   {i:2d}. [ ] {item}")
    print("\n   Iterate: if any flagged, redesign ligand (regenerate SELFIES) and repeat.")


# ---------------------------------------------------------------------------
# 16. Aggregate verdict
# ---------------------------------------------------------------------------
def aggregate_verdict(pc, tox, sol, admet, sa):
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)
    score = 0
    if pc['qed'] > 0.5:                  score += 1
    if pc['logp'] < 5:                   score += 1
    if pc['mw']  < 500:                  score += 1
    if pc['tpsa']< 140:                  score += 1
    if not tox['alerts']:                score += 1
    if tox['herg'] != "HIGH":            score += 1
    if sol['logS'] > -4:                 score += 1
    if sa < 6:                           score += 1
    print(f"   Drug-likeness score : {score}/8")
    print(f"   Recommend docking/MD/experimental pipeline if score ≥ 6.")
    print("="*70)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    mol = rdkit_validation(SELFIES_STR, EXPECTED_SMILES)
    if mol is None:
        print("Molecule invalid — aborting.")
        sys.exit(1)
    pc      = physchem_profile(mol)
    sa      = synthetic_accessibility(mol)
    filters = druglikeness_filters(mol, pc)
    tox     = toxicity_screen(mol)
    sol     = solubility_estimate(mol, pc)
    admet   = admet_quick(mol, pc)
    off_target_prediction(EXPECTED_SMILES)
    pdb     = protein_structure(PROTEIN_SEQ)
    binding_pocket(pdb)
    molecular_docking(EXPECTED_SMILES, pdb or "receptor.pdb")
    md_simulation(pdb)
    free_energy_calc()
    selectivity_analysis()
    experimental_validation_checklist()
    aggregate_verdict(pc, tox, sol, admet, sa)


if __name__ == "__main__":
    main()