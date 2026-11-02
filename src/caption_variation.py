import os
import time
import base64
from pathlib import Path
from typing import Dict, Any
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"

API_KEY = "key"
RWTH_GPT_ENDPOINT = "https://chat.kiconnect.nrw/api/v1/chat/completions"
VISION_MODEL_NAME = "gpt-5.4-mini" 
TEXT_MODEL_NAME = "gpt-oss-120b" 
DATASET_DIR = DATA_ROOT / "nature_data" / "V1 UAV-RSOD_Dataset for Segmentation"
ORIGINAL_IMAGE_FOLDER = DATASET_DIR / "1 Images" #original images from UAV-RSOD dataset
EXCEL_FILE = DATASET_DIR / "classes.xlsx" # Excel file containing class information, manually created
PROMPT_FOLDER = DATASET_DIR / "1 Prompts" # Folder to save the generated prompts
CLASSES = ["Jerrycan", "Paperbox", "Person", "TrafficCone", "IronRod", "Boulder"]
WEATHER_CONDITIONS = ["fog", "rain", "sunlight", "evening", "normal"]
CAMERA_PERSPECTIVES = ["nadir", "low_fpv", "oblique", "normal"] 

# --- Phase 1: Vision Prompt ---
BASE_VISION_PROMPT = (
    "Analyze this UAV image of railway infrastructure. "
    "I ONLY need you to describe the static background environment. "
    "Describe the physical structure of the tracks (e.g., rigid parallel steel rails, straight or curving) "
    "and the base (e.g., grey gravel ballast, wooden/concrete sleepers). Detail their spatial arrangement. "
    "Describe the immediate surrounding terrain. "
    "CRITICAL: Do NOT describe any obstacles, do NOT describe the weather or lighting, and do NOT describe the camera angle. "
    "Output a single, factual paragraph."
)

def extract_base_geometry(image_path: Path) -> str:
    """
    Calls the Vision model ONCE per image to extract the static track geometry.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    data = {
        "model": VISION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASE_VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }

    resp = requests.post(RWTH_GPT_ENDPOINT, headers=headers, json=data, timeout=120)
    resp.raise_for_status()

    result: Dict[str, Any] = resp.json()
    return result["choices"][0]["message"]["content"].strip()


# --- Phase 2: Text Blending Prompt ---
def build_blending_prompt(base_geometry: str, weather: str, target_class: str, perspective: str) -> str:
    """
    Builds a text-only prompt asking the LLM to rewrite the base geometry with new conditions.
    Handles the "Empty" class to ensure no obstacles are hallucinated.
    """
    perspective_instructions = {
        "normal": "The scene is captured from the standard drone angle present in the original data.",
        "nadir": "The scene is viewed strictly as a top-down nadir view from a medium-altitude drone (looking directly down).",
        "low_fpv": "The scene is viewed as a low-altitude FPV drone shot, flying extremely close to the ground, looking directly down the tracks.",
        "oblique": "The scene is viewed as a steep oblique aerial view, capturing depth and the tracks from a distinct elevated side-angle."
    }

    weather_instructions = {
        "normal": "Standard daylight conditions.",
        "rain": "Heavy rain affecting the scene: wet, reflective surfaces, raindrops, reduced visibility.",
        "fog": "Heavy fog: reduced visibility, muted colors, softened edges, creating a sense of obscurity.",
        "sunlight": "Bright sunlight: strong shadows, glare on the rails, high color saturation.",
        "evening": "Late evening with spare light: long shadows, muted colors, dim conditions."
    }

    if target_class == "Empty":
        obstacle_instruction = "3. Obstacle: There is NO obstacle in the scene. The railway tracks are completely empty and clear. Emphasize the clear, uninterrupted path of the rails."
    else:
        obstacle_instruction = (
            f"3. Obstacle: There is a {target_class} acting as an obstacle. Invent a plausible exact spatial relationship "
            f"(e.g., lying across the rails, wedged in the gravel) and describe its color/material blending into the {weather} environment."
        )

    system_prompt = (
        "You are an expert computer vision annotator generating dataset captions for text-to-image AI training. "
        "Your task is to take a base description of railway tracks and rewrite it into a single, highly detailed, dense paragraph. "
        "Do not use introductory phrases like 'This image shows'. Do not use bullet points or formatting."
    )

    user_prompt = (
        f"Here is the base geometry of the tracks: '{base_geometry}'\n\n"
        "Rewrite this scene seamlessly incorporating the following three elements:\n"
        f"1. Camera Perspective: {perspective_instructions[perspective]}\n"
        f"2. Environment/Lighting: {weather_instructions[weather]}\n"
        f"{obstacle_instruction}\n\n"
        "Output ONLY the final descriptive paragraph."
    )
    
    return system_prompt, user_prompt

def blend_synthetic_caption(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the Text-Only LLM to weave the permutations together.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": TEXT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    resp = requests.post(RWTH_GPT_ENDPOINT, headers=headers, json=data, timeout=60)
    resp.raise_for_status()

    result: Dict[str, Any] = resp.json()
    return result["choices"][0]["message"]["content"].strip()


def process_images() -> None:
    folder_path = Path(ORIGINAL_IMAGE_FOLDER)
    prompt_dir = Path(PROMPT_FOLDER)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    if not folder_path.is_dir() or not Path(EXCEL_FILE).is_file():
        print("Missing image folder or excel file.")
        return
        
    df = pd.read_excel(EXCEL_FILE, header=None)
    classes_list = df[0].tolist() 

    image_files = list(folder_path.glob("*.[jJ][pP][gG]")) + list(folder_path.glob("*.[jJ][pP][eE][gG]"))
    
    print(f"Found {len(image_files)} image(s) to process. Starting Phase 1 & 2 pipeline…\n")

    for img_path in image_files:
        try:
            file_number = int(img_path.stem)
        except ValueError:
            continue
            
        if file_number < 1 or file_number > len(classes_list):
            continue

        # --- Phase 1: Get or Cache Base Geometry ---
        base_text_path = prompt_dir / f"base_geometry_{file_number}.txt"
        
        if base_text_path.exists():
            base_geometry = base_text_path.read_text(encoding="utf-8")
        else:
            print(f"[{img_path.name}] Extracting base track geometry via Vision API...", end=" ", flush=True)
            try:
                base_geometry = extract_base_geometry(img_path)
                base_text_path.write_text(base_geometry, encoding="utf-8")
            except Exception as exc:
                print(f"VLM Error → {exc}")
                continue

        target_classes = []
        target_classes.append(classes_list[file_number - 1])
        target_classes = list(dict.fromkeys(target_classes))

        # --- Phase 2: Text Permutations ---
        print(f"[{img_path.name}] Generating variations for classes: {CLASSES}...")

        for target_class in CLASSES:
            for weather in WEATHER_CONDITIONS:
                for perspective in CAMERA_PERSPECTIVES:                    
                    clean_class_name = target_class.replace(' ', '')
                    new_name = f"{weather}_{perspective}_{clean_class_name}_{file_number}.txt"
                    txt_path = prompt_dir / new_name

                    if txt_path.exists():
                        print("Already exists, skipping:", new_name)
                        continue

                    try:
                        sys_p, usr_p = build_blending_prompt(base_geometry, weather, target_class, perspective)
                        caption = blend_synthetic_caption(sys_p, usr_p)
                        txt_path.write_text(caption, encoding="utf-8")
                    except Exception as exc:
                        print(f"LLM Error on {weather}|{perspective}|{target_class} → {exc}")

    print("\nBatch captioning finished!")


if __name__ == "__main__":
    process_images()