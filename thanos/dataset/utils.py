import json
import os

def get_dataset_root():
    thanos_root = os.path.join(os.path.dirname(__file__), "..", "..")
    config_path = os.path.join(thanos_root, "dataset_config.json")
    with open(config_path) as f:
        dataset_root = json.load(f)["ipn"]
    return dataset_root