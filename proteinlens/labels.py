import numpy as np
import requests


def get_quality_label(pdb_id):
    """Assigns good/medium/bad label from RCSB crystallographic validation metrics."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[WARN] Could not fetch validation data for {pdb_id}")
        return None, None

    data = response.json()

    try:
        geom = data["pdbx_vrpt_summary_geometry"]
        if isinstance(geom, list):
            geom = geom[0]

        diff = data.get("pdbx_vrpt_summary_diffraction", {})
        if isinstance(diff, list):
            diff = diff[0]

        clashscore       = geom.get("clashscore")
        rama_outliers    = geom.get("percent_ramachandran_outliers")
        rotamer_outliers = geom.get("percent_rotamer_outliers")
        rsrz_outliers    = diff.get("percent_rsrzoutliers")

        scores = []
        if clashscore is not None:
            scores.append(2 if clashscore < 10 else (1 if clashscore < 25 else 0))
        if rama_outliers is not None:
            scores.append(2 if rama_outliers < 0.5 else (1 if rama_outliers < 2.0 else 0))
        if rotamer_outliers is not None:
            scores.append(2 if rotamer_outliers < 1.0 else (1 if rotamer_outliers < 5.0 else 0))
        if rsrz_outliers is not None:
            scores.append(2 if rsrz_outliers < 5.0 else (1 if rsrz_outliers < 10 else 0))

        if len(scores) < 2:
            print(f"[WARN] Too few valid metrics for {pdb_id}, skipping")
            return None, None

        composite = np.mean(scores)
        label = "good" if composite >= 1.5 else ("medium" if composite >= 0.75 else "bad")
        return composite, label

    except KeyError as e:
        print(f"[WARN] Missing validation field for {pdb_id}: {e}")
        return None, None