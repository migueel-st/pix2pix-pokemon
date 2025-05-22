"""Module for preparing the Pokemon dataset for paired image generation tasks."""

import os
import csv
from pathlib import Path
import cv2
import numpy as np
from cv2.typing import MatLike

# TODO: move this to a config file
type_to_color = {
    "bug": (0, 255, 150),
    "dark": (0, 0, 0),
    "dragon": (0, 0, 255),
    "electric": (255, 255, 0),
    "fairy": (255, 182, 193),
    "fighting": (255, 0, 0),
    "fire": (255, 150, 0),
    "flying": (135, 206, 250),
    "ghost": (128, 0, 128),
    "grass": (0, 255, 0),
    "ground": (139, 69, 19),
    "ice": (173, 216, 230),
    "normal": (50, 50, 50),
    "poison": (128, 0, 128),
    "psychic": (255, 20, 147),
    "rock": (139, 69, 19),
    "steel": (192, 192, 192),
    "water": (0, 150, 255),
}


def split_pokemon_dataset_by_type(dataset_path: Path):
    """Split the Pokemon dataset into subdirectories based on their types."""
    images_path = dataset_path / "images"
    print(f"Preparing dataset at {dataset_path} ...")
    with open(dataset_path / "pokemon.csv", newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            name = row['Name'].strip()
            main_type = row['Type1'].strip().lower()
            if os.path.exists(images_path / f"{name}.png"):
                type_dir = dataset_path / "types" / main_type
                type_dir.mkdir(parents=True, exist_ok=True)
                convert_transparent_to_white(str(images_path / f"{name}.png"),
                                             str(type_dir / f"{name}.png"))
    print("Dataset prepared ✅")
    print("Images are organized by type in the '<dataset_path>/types' directory.")


def convert_transparent_to_white(input_path: str, output_path: str):
    """Convert transparent PNG images to white background JPG images."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)    
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    # New image with white background
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(output_path, new_img)


def create_type_masks(dataset_path: Path):
    """Create color masks for each type in the dataset."""
    types_path = dataset_path / "types"
    for type_dir in types_path.iterdir():
        for image_path in type_dir.iterdir():
            binary = segment_pokemon_threshold(str(image_path))
            color = tuple(type_to_color[type_dir.name])
            colored_mask = refine_with_contours(binary, color)
            cv2.imwrite(str(type_dir / f"{image_path.stem}_mask.png"), colored_mask)


def refine_with_contours(mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    blank_mask: MatLike = cv2.cvtColor(np.full_like(mask,  255), cv2.COLOR_GRAY2BGR)
    refined_mask = cv2.drawContours(blank_mask,
                                    contours=[largest_contour],
                                    contourIdx=-1,
                                    color=color,
                                    thickness=cv2.FILLED)
    return cv2.cvtColor(refined_mask, cv2.COLOR_BGR2RGB)


def segment_pokemon_threshold(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    return binary

if __name__ == "__main__":
    # TODO: Add this as a command line argument
    dataset_path = Path("prepare") / "pokemon-images-and-types"
    split_pokemon_dataset_by_type(dataset_path)
    create_type_masks(dataset_path)
    # concat_image_and_masks(dataset_path)