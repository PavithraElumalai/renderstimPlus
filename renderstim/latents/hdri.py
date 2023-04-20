import kubric as kb
import os
import pathlib
import json

def get_hdri_id(rng):
    with open(os.path.join(pathlib.Path().resolve().parent,"HDRI_haven.json")) as f:
        hdri_assets = json.load(f)['assets']
    hdri_ids = list(hdri_assets.keys())
    hdri_id = rng.choice(hdri_ids)
    
    return hdri_id