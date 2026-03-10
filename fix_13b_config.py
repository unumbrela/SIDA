"""
Fix SIDA-13B-description config.json: change model_type from 'llava' to 'llava_sida'
for compatibility with transformers 5.x.
Run this on the server before training.
"""
import json
import sys

config_path = sys.argv[1] if len(sys.argv) > 1 else "./ck/SIDA-13B-description/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

old_type = config.get("model_type", "")
if old_type != "llava_sida":
    config["model_type"] = "llava_sida"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Fixed: model_type '{old_type}' -> 'llava_sida' in {config_path}")
else:
    print(f"Already correct: model_type = 'llava_sida' in {config_path}")
