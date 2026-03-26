from Bio.PDB import MMCIFParser
from config import PDB_DIR
import numpy as np
import requests
import os


def fetch_and_parse(pdb_id):
    """Download and parse a mmCIF file from RCSB. Returns (structure, error)."""
    pdb_id = pdb_id.lower()
    os.makedirs(PDB_DIR, exist_ok=True)
    path = f"{PDB_DIR}/{pdb_id}.cif"
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None, f"Could not download {pdb_id.upper()} (status {response.status_code})"
        with open(path, "w") as f:
            f.write(response.text)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, path)
    return structure, None


def get_mean_bfactor(structure):
    """Mean B-factor — measures average atomic displacement."""
    b = [a.bfactor for m in structure for c in m
         for r in c if r.id[0] == " " for a in r]
    return np.mean(b) if b else np.nan


def get_bfactor_std(structure):
    """Std deviation of B-factors — high variance = flexible regions."""
    b = [a.bfactor for m in structure for c in m
         for r in c if r.id[0] == " " for a in r]
    return np.std(b) if b else np.nan


def get_hydrophobic_ratio(structure):
    """Fraction of hydrophobic residues — core packing drives stability."""
    hydrophobic = {"LEU", "VAL", "ILE", "PHE", "MET", "TRP", "PRO", "ALA"}
    res = [r.resname for m in structure for c in m
           for r in c if r.id[0] == " "]
    return sum(1 for r in res if r in hydrophobic) / len(res) if res else np.nan


def get_charged_ratio(structure):
    """Fraction of charged residues — salt bridges contribute to stability."""
    charged = {"ARG", "LYS", "ASP", "GLU", "HIS"}
    res = [r.resname for m in structure for c in m
           for r in c if r.id[0] == " "]
    return sum(1 for r in res if r in charged) / len(res) if res else np.nan


def get_avg_chain_length(structure):
    """Average chain length — larger proteins have more stable cores."""
    lengths = [sum(1 for r in c if r.id[0] == " ")
               for m in structure for c in m]
    return np.mean(lengths) if lengths else np.nan


def extract_features(pdb_id):
    """Extract all features for a given PDB ID. Downloads if not cached."""
    path = f"{PDB_DIR}/{pdb_id.lower()}.cif"

    # Download if not already cached
    if not os.path.exists(path):
        _, err = fetch_and_parse(pdb_id)
        if err:
            print(f"[WARN] Could not download {pdb_id}: {err}")
            return None

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id.lower(), path)
    return {
        "pdb_id":           pdb_id,
        "mean_bfactor":     get_mean_bfactor(structure),
        "bfactor_std":      get_bfactor_std(structure),
        "hydro_ratio":      get_hydrophobic_ratio(structure),
        "charged_ratio":    get_charged_ratio(structure),
        "avg_chain_length": get_avg_chain_length(structure),
    }