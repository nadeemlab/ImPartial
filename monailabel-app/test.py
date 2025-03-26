import os
import json
from pathlib import Path
import sys 
import requests
from PIL import Image

from dataprocessing.utils import rois_to_mask

sys.path.append("../../")
from impartial.general.evaluation import get_performance

BASE_URL = "http://localhost:8000"
MODEL_NAME = "full_labels"
IMAGES_PATH = "/data/DAPI1CH"
LABELS_PATH = "/data/DAPI1CH/labels"


def infer(image_id):
    print(f"infer {image_id}")
    res = requests.post(
        f"{BASE_URL}/infer/impartial",
        params={
            "image": image_id,
            "output": "image"
        }
    )
    res.raise_for_status()

    return res.content


def is_png(path):
    return os.path.splitext(path)[1].lower() == ".png"


def get_root(path):
    return os.path.splitext(path)[0]


if __name__ == '__main__':
    outputs_path = os.path.join(IMAGES_PATH, "outputs", MODEL_NAME)
    # make user the output directory exists
    Path(outputs_path).mkdir(exist_ok=True)

    image_names = sorted(filter(is_png, os.listdir(IMAGES_PATH)))
    images = [Image.open(os.path.join(IMAGES_PATH, i)) for i in image_names]

    image_ids = [get_root(i) for i in image_names]
    image_sizes = [i.size for i in images]

    labels = [
        rois_to_mask(os.path.join(LABELS_PATH, f"{i}.zip"), s)
        for i, s in zip(image_ids, image_sizes)
    ]

    outputs = []
    for i, s in zip(image_ids, image_sizes):
        zip_file_content = infer(i)

        output_path = os.path.join(outputs_path, f"{i}.zip")
        with open(output_path, "wb") as f:
            f.write(zip_file_content)

        outputs.append(rois_to_mask(output_path, s))

    metrics = [get_performance(l, o) for l, o in zip(labels, outputs)]

    with open(os.path.join(outputs_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
