import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"
OUTPUT_DIR = DATA_ROOT / "rule_based_prompts"
NUMBER_OF_BASE_SCENES = 315
CLASSES = ["Jerrycan", "Boulder", "Branch", "IronRod", "Barrel", "Person"] 
WEATHERS = ["fog", "rain", "sunlight", "evening", "normal"]
PERSPECTIVES = ["nadir", "low_fpv", "oblique", "normal"]

# 1. Camera Perspectives
PERSPECTIVE_TEMPLATES = {
    "nadir": "Top-down nadir aerial view from a medium-altitude drone looking directly down at the tracks.",
    "low_fpv": "Low-altitude FPV drone perspective flying close to the ground, looking forward directly down the railway.",
    "oblique": "Steep oblique aerial drone view capturing the tracks, depth, and environment from a distinct elevated side-angle.",
    "normal": "Standard medium-altitude aerial drone view of the railway infrastructure."
}

# 2. Track Geometry Variations
TRACK_TEMPLATES = [
    "The tracks feature rigid parallel steel rails secured to heavy concrete sleepers, resting on a bed of dark grey gravel ballast.",
    "Straight parallel steel rails are laid across aged, weathered wooden sleepers embedded in light grey gravel.",
    "Gently curving steel railway tracks sit atop evenly spaced concrete sleepers, surrounded by scattered, rusty ballast stones."
]

# 3. Terrain Variations
TERRAIN_TEMPLATES = [
    "The immediate surrounding terrain consists of sparse green vegetation and dry dirt.",
    "Thick green shrubs and overgrown grass heavily border the sides of the railway.",
    "The tracks cut through a barren, rocky landscape with patches of brown earth."
]

# 4. Obstacle Attributes
OBSTACLE_ATTRIBUTES = {
    "Jerrycan": ["a bright red plastic jerrycan with a handle", "a large, dented metal fuel canister", "a sturdy blue plastic container"],
    "Boulder": ["a large, jagged grey boulder", "a heavy, irregularly shaped rock", "a massive stone with rough, uneven surfaces"],
    "Branch": ["a thick, broken tree branch with rough bark", "a long, twisted wooden limb", "a large, fallen tree limb with scattered leaves"],
    "IronRod": ["a long, rusty iron rod", "a thick steel bar with a rough, corroded surface", "a heavy metal pipe with visible rust"],
    "Barrel": ["a large, dented metal barrel", "a rusty cylindrical container", "a heavy steel drum with a weathered surface"],
    "Person": ["a person wearing a bright orange safety vest and hard hat", "a figure in dark clothing crouched near the tracks", "a worker in a yellow high-visibility jacket standing on the railway"]
}

# 5. Spatial Relationships
SPATIAL_RELATIONS = [
    "lying horizontally across both steel rails",
    "resting precariously on the outer left sleeper",
    "wedged deeply in the center gravel between the rails",
    "sitting directly on the right steel rail"
]

# 6. Weather & Lighting Conditions
WEATHER_TEMPLATES = {
    "normal": "Standard daylight conditions provide even lighting across the scene with soft, natural shadows.",
    "rain": "Heavy rain pours over the scene, creating wet, highly reflective surfaces on the steel rails and visible raindrops in the air.",
    "fog": "Heavy fog blankets the scene, resulting in reduced visibility, muted colors, and softened edges that create a sense of obscurity.",
    "sunlight": "Bright, harsh sunlight illuminates the scene, casting strong dark shadows and creating a bright glare on the steel rails.",
    "evening": "Late evening light sets over the scene, casting long, dramatic shadows and reducing overall visibility with dim, muted colors."
}

def generate_rule_based_prompt(perspective, track, terrain, obstacle_desc, spatial, weather):
    """
    Concatenates the selected templates into a single dense paragraph.
    """
    return (
        f"{PERSPECTIVE_TEMPLATES[perspective]} "
        f"{track} {terrain} "
        f"There is an obstacle present: {obstacle_desc} is {spatial}. "
        f"{WEATHER_TEMPLATES[weather]} "
        f"The scene is dense with visual facts and spatial anchors."
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate total combinations
    total_permutations = NUMBER_OF_BASE_SCENES * len(CLASSES) * len(WEATHERS) * len(PERSPECTIVES)
    print(f"Generating {total_permutations} rule-based prompts...\n")
    
    prompt_count = 0
    
    for scene_id in range(1, NUMBER_OF_BASE_SCENES + 1):
        base_track = random.choice(TRACK_TEMPLATES)
        base_terrain = random.choice(TERRAIN_TEMPLATES)
        
        for injected_class in CLASSES:
            clean_class = injected_class.replace(' ', '')
            
            for weather in WEATHERS:
                for perspective in PERSPECTIVES:
                    obstacle_desc = random.choice(OBSTACLE_ATTRIBUTES[injected_class])
                    spatial_rel = random.choice(SPATIAL_RELATIONS)
                    
                    # Build the prompt string
                    prompt_text = generate_rule_based_prompt(
                        perspective=perspective,
                        track=base_track,
                        terrain=base_terrain,
                        obstacle_desc=obstacle_desc,
                        spatial=spatial_rel,
                        weather=weather
                    )
                    
                    filename = f"{weather}_{perspective}_{clean_class}_{scene_id}.txt"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(prompt_text)
                    
                    prompt_count += 1

if __name__ == "__main__":
    main()