import os

import numpy as np
from PIL import Image


def normalize(xs):
    return (255.0 * (xs - xs.min())) / (xs.max() - xs.min())


def array_to_pil(xs):
    return Image.fromarray(np.uint8(normalize(xs)), 'L')


def get_image(npz_file_path, mode="first"):
    with open(npz_file_path, "rb") as f:
        if "image" not in np.load(f):
            print("a")
        image = np.load(f)["image"]

    channels = [array_to_pil(ch) for ch in np.rollaxis(image, 2)]

    if mode == "first":
        return channels[0]
    elif mode == "second":
        return channels[1]
    else:
        channels[0].paste(channels[1], (0, 0), channels[1])
        return channels[0]


def get_scribbles(npz_file_path, mode="first"):
    with open(npz_file_path, "rb") as f:
        scribble = np.load(f)["scribble"]

    channels = [array_to_pil(ch) for ch in np.rollaxis(scribble, 2)]

    if mode == "first":
        return Image.merge("RGB", (channels[0], channels[1], channels[1]))
    elif mode == "second":
        return channels[1]
    else:
        channels[0].paste(channels[1], (0, 0), channels[1])
        return channels[0]


def migrate_to_monai(dataset_path):

    new_dataset_path = os.path.join(
        os.path.dirname(dataset_path),
        f"{os.path.split(dataset_path)[-1]}_new"
    )

    os.makedirs(new_dataset_path, exist_ok=True)

    def is_npz(x):
        return os.path.isfile(x) and os.path.splitext(x)[1] == ".npz"

    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    npz_files = [f for f in files if is_npz(f)]

    def new_path(p):
        return os.path.join(
            new_dataset_path, f"{os.path.splitext(os.path.split(p)[-1])[0]}.png"
        )

    images = [
        {
            "path": new_path(f),
            "image": get_image(f, mode="both"),
        }
        for f in npz_files if "scribble" not in f and "label" not in f
    ]

    for i in images:
        i["image"].save(i["path"])


if __name__ == '__main__':
    migrate_to_monai("./Data/cellpose")
