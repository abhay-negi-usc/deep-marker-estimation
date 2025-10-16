# ~/isaacsim/python.sh synthetic_data_generation/marker_obj_sdg_lights.py 
# ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh data_generation/multi_marker_data_generation.py 

# DESCRIPTION: 
# Multi-marker synthetic data generation script with:
# - Multiple markers spawned at randomized poses in the scene
# - Marker IDs saved in metadata for all markers
# - Segmentation coloring based on marker ID (each marker gets distinct color)
# - RGB images output in black and white (grayscale)
# - Randomization of: marker poses, background plane textures, lighting direction, marker textures

# IMPORTS 
import argparse
import json
import os
import colorsys
import gc
import psutil

import yaml
from isaacsim import SimulationApp
import time 
import asyncio
from PIL import Image
import numpy as np 

from scipy.spatial.transform import Rotation as R 

import carb
import carb.settings

import random
from itertools import chain

import sys  
import re 

#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SET UP DIRECTORIES 
timestr = time.strftime("%Y%m%d-%H%M%S") 
print(os.getcwd())
if os.getcwd() == '/home/anegi/abhay_ws/deep-marker-estimation': # isaac machine 
    OUT_DIR = os.path.join(os.getcwd(), "data_generation", "multi_marker_output", "sdg_markers_" + timestr)
    dir_textures = "/home/anegi/abhay_ws/deep-marker-estimation/data_generation/assets/tag36h11_no_border_16/"
    sys.path.append("/home/anegi/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
    # dir_backgrounds = "/media/anegi/easystore/abhay_ws/marker_detection_failure_recovery/background_images" 
    dir_backgrounds = "/home/anegi/Downloads/test2017" 
else: # CAM machine 
    OUT_DIR = os.path.join("/media/rp/Elements1/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/", "output", "sdg_markers_" + timestr)
    dir_textures = "/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/sdg_tag"
    sys.path.append("/home/rp/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
    dir_backgrounds = "/media/rp/Elements1/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/background_images" 

# dir_textures = "./synthetic_data_generation/assets/tags/aruco dictionary 6x6 png" 

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"rgb"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"seg"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"pose"), exist_ok=True) 
os.makedirs(os.path.join(OUT_DIR,"metadata"), exist_ok=True) 
tag_textures = [os.path.join(dir_textures, f) for f in os.listdir(dir_textures) if os.path.isfile(os.path.join(dir_textures, f))] 
print("Set up directories. OUT_DIR: ", OUT_DIR)
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# CONFIG 
config = {
    "launch_config": {
        "renderer": "RayTracedLighting", # RayTracedLighting, PathTracing
        "headless": True,
    },
    "env_url": "",
    "working_area_size": (1,1,10),
    "rt_subframes": 4,
    "num_frames": 100_000,
    "num_cameras": 1,
    "camera_collider_radius": 0.5,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 1.0,
    "resolution": (960, 600),
    "camera_properties_kwargs": {
        "focalLength": 12.5,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": (0., 10000),
        "cameraNearFar": (0.0001, 10000),
    },
    "camera_look_at_target_offset": 0.25,  
    "camera_distance_to_target_min_max": (0.100, 1.000),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": OUT_DIR,
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
        # "semantic_segmentation": True,  
        # "colorize_semantic_segmentation": True,
    },
    "num_markers": 16,  # Maximum number of markers to use from the pool (num_marker_patterns)
    "num_markers_scene": 16,  # Exact number of markers to spawn in each scene iteration
    "marker_distance_range": (0.1, 2.0),  # Distance range for markers from camera
    "marker_horizontal_range": (-1.0, 1.0),  # Horizontal position range
    "marker_vertical_range": (-1.0, 1.0),  # Vertical position range

    "lights": "distant_light", # dome, distant_light 
}
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# START SIMULATION APP FOR IMPORTS 
# Isaac nucleus assets root path
stage = None
launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)
print("Simulation app started.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SYS DEPENDENT IMPORTS 
import object_based_sdg_utils  
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt

# from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.nucleus import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics
from pxr import Usd, UsdShade, Gf
#------------------------------------------------------------------------------------------------------------------------------------------------------#


# HELPER FUNCTIONS 
# TODO: export to a separate file 

assets_root_path = get_assets_root_path() # out of place here but needs to be after its import 

# Add transformation properties to the prim (if not already present) 

# Global profiling variables (simplified)
profile_data = {}
frame_times = []

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def print_memory_status(frame_num):
    """Print memory usage status"""
    memory_mb = get_memory_usage()
    print(f"[SDG] Frame {frame_num}: Memory={memory_mb:.1f}MB")



# Capture motion blur by combining the number of pathtraced subframes samples simulated for the given duration
def capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=64, apply_blur=True):
    # For small step sizes the physics FPS needs to be temporarily increased to provide movements every syb sample
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    if apply_blur: 
        # Enable motion blur (if not enabled)
        is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
        if not is_motion_blur_enabled:
            carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
        # Number of sub samples to render for motion blur in PathTracing mode
        carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False, rt_subframes=3)

    # Restore the original physics FPS
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    # Restore the previous render and motion blur  settings
    if apply_blur: 
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)

def get_world_transform_xform_as_np_tf(prim: Usd.Prim):
    """
    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)

    return np.array(world_transform).transpose()

# Util function to save rgb annotator data (converted to black and white)
def write_rgb_data(rgb_data, file_path):
    rgb_img = Image.fromarray(rgb_data, "RGBA")
    # Convert to grayscale (black and white)
    grayscale_img = rgb_img.convert('L')
    # Convert back to RGBA for consistency
    bw_rgba = Image.new('RGBA', grayscale_img.size)
    bw_rgba.paste(grayscale_img)
    bw_rgba.save(file_path + ".png")

# Util function to save semantic segmentation annotator data with marker-specific coloring
def write_sem_data(sem_data, file_path, marker_colors=None):
    id_to_labels = sem_data["info"]["idToLabels"]
    
    # Create enhanced JSON data with color information
    enhanced_labels = {}
    for seg_id_str, seg_label_data in id_to_labels.items():
        # Handle different formats of seg_label_data
        if isinstance(seg_label_data, dict):
            # If it's a dict, extract the class name
            seg_label = seg_label_data.get("class", seg_label_data.get("label", "unknown"))
        else:
            # If it's a string, use it directly
            seg_label = seg_label_data
        
        # Add color information if available
        color_info = None
        if marker_colors is not None and seg_label in marker_colors:
            color_info = marker_colors[seg_label][:3]  # RGB only for JSON
        elif seg_label == "background":
            color_info = [50, 50, 50]
        else:
            color_info = [128, 128, 128]
        
        enhanced_labels[seg_id_str] = {
            "label": seg_label,
            "color": color_info
        }
    
    with open(file_path + ".json", "w") as f:
        json.dump(enhanced_labels, f, indent=2)
    
    sem_image_data = np.frombuffer(sem_data["data"], dtype=np.uint8).reshape(*sem_data["data"].shape, -1)
    
    # If marker colors are provided, apply custom coloring
    if marker_colors is not None:
        # Create a new image with custom colors
        colored_sem_data = np.zeros_like(sem_image_data)
        
        # Map each segmentation ID to the corresponding custom color
        for seg_id_str, seg_label_data in id_to_labels.items():
            seg_id = int(seg_id_str)
            
            # Handle different formats of seg_label_data
            if isinstance(seg_label_data, dict):
                # If it's a dict, extract the class name
                seg_label = seg_label_data.get("class", seg_label_data.get("label", "unknown"))
            else:
                # If it's a string, use it directly
                seg_label = seg_label_data
            
            # Find all pixels with this segmentation ID
            # The segmentation data uses the first channel for the ID
            mask = sem_image_data[:, :, 0] == seg_id
            
            if seg_label in marker_colors:
                # Use custom color for this marker
                color = marker_colors[seg_label]
                colored_sem_data[mask] = color[:4]  # RGBA
            else:
                # Use a default color for non-marker objects (like background)
                if seg_label == "background":
                    colored_sem_data[mask] = [50, 50, 50, 255]  # Dark gray for background
                else:
                    colored_sem_data[mask] = [128, 128, 128, 255]  # Gray for other objects
        
        sem_image_data = colored_sem_data
    
    sem_img = Image.fromarray(sem_image_data, "RGBA")
    sem_img.save(file_path + ".png")

def extract_marker_id_from_texture_path(texture_path):
    """Extract marker ID from texture filename like 'tag36_11_00005.png'"""
    # Handle both full paths and just filenames
    filename = os.path.basename(texture_path)
    match = re.search(r"tag36_11_(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return 0



def write_pose_data(pose_data, file_path):
    with open(file_path + ".json", "w") as f:
        json.dump(pose_data, f) 

def write_metadata(metadata, file_path): 
    with open(file_path + ".json", "w") as f:
        json.dump(metadata, f) 

def serialize_vec3f(vec3f):
    # Convert Gf.Vec3f to a list or dictionary
    return [vec3f[0], vec3f[1], vec3f[2]]

# Update the app until a given simulation duration has passed (simplified)
def run_simulation_loop(duration):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()



#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SET UP ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
    # Remove any previous semantics in the loaded stage
    for prim in stage.Traverse():
        # remove_all_semantics(prim)  # Commented out - function not available
        pass
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (2, 2, 2))
working_area_min = (working_area_size[0] / -2, working_area_size[1] / -2, -working_area_size[2])
working_area_max = (working_area_size[0] / 2, working_area_size[1] / 2, 0)
# Create a physics scene to add or modify custom physics settings
usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
if physics_scenes:
    physics_scene = physics_scenes[0]
else:
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)
rep.orchestrator.set_capture_on_play(False)
print("Environment set up.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# CAMERA 
cam = rep.create.camera(
    position=(0,0,0), 
    rotation=(0,-90,270), 
) 
rp_cam = rep.create.render_product(cam, config.get("resolution", (960, 600))) 
cam_prim = cam.get_output_prims()["prims"][0] 
camera = [cam]
render_products = [rp_cam]
num_cameras = config["num_cameras"] # NOTE: placeholder for now because only using 1 cam 
cam_cam_prim = cam_prim.GetChildren()[0] 
cam_cam_prim.GetAttribute("clippingRange").Set((0.0001, 1000000)) 
print("Camera set up.")

#------------------------------------------------------------------------------------------------------------------------------------------------------#

# LIGHTS 
if config["lights"] == "dome": 
    print("Applying dome light.") 
    dome_light = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight") 
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
elif config["lights"] == "distant_light": 
    # rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
    #     loc_min=working_area_min, loc_max=working_area_max, scale_min_max=(1,1)
    # )
    distant_light = rep.create.light(
        light_type="distant",
        # color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        color=(1, 1, 1),
        # temperature=rep.distribution.normal(6500, 500),
        intensity=1.0, 
        exposure=rep.distribution.uniform(10, 16), 
        rotation=rep.distribution.uniform((-180,-180,-180), (180,180,180)),
        position=(0,0,3),
        count=1,
        # color_temperature=rep.distribution.uniform(2500, 10000),
    )
    distant_light_prim = distant_light.get_output_prims()["prims"][0] 
    distant_light_lighting_prim = distant_light_prim.GetChildren()[0]

    # FIXME: REVERT IF NOT REQUIRED 
    dome_light = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight") 
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(100.0)
print("Lights set up.")

#------------------------------------------------------------------------------------------------------------------------------------------------------#

# MARKERS 
# Create exactly num_markers_scene markers that will be used in each iteration
num_markers_scene = config.get("num_markers_scene", 8)
num_marker_patterns = len(tag_textures)  # Total number of available texture patterns
num_markers = num_markers_scene  # Create exactly the number of markers we'll show
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []
markers = []
marker_prims = []
marker_textures_sequences = []

# Define distinct colors for each possible marker pattern (not just active markers)
marker_colors_by_texture_id = {}
base_colors = [
    [255, 0, 0, 255],    # Red
    [0, 255, 0, 255],    # Green  
    [0, 0, 255, 255],    # Blue
    [255, 255, 0, 255],  # Yellow
    [255, 0, 255, 255],  # Magenta
    [0, 255, 255, 255],  # Cyan
    [255, 128, 0, 255],  # Orange
    [128, 0, 255, 255],  # Purple
]

# Extend colors if we have more texture patterns than base colors
while len(base_colors) < num_marker_patterns:
    # Generate additional colors using golden angle for good distribution
    hue = (len(base_colors) * 137.5) % 360  # Golden angle for color distribution
    rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 1.0)
    base_colors.append([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 255])

# Assign colors to texture IDs
for texture_path in tag_textures:
    texture_id = extract_marker_id_from_texture_path(texture_path)
    color_idx = texture_id % len(base_colors)
    marker_colors_by_texture_id[texture_id] = base_colors[color_idx]

# Create marker pool for the scene
for marker_idx in range(num_markers):
    label = f"tag{marker_idx}"
    
    # Create the marker plane
    marker = rep.create.plane(
        position=(0, 0, -1.0),
        scale=(0.05, 0.05, 0.05), 
        rotation=(0, 0, 0),   
        name=f"marker_{marker_idx}",  # Use clean naming without "tag" prefix
        semantics=[("class", label)],
    )
    
    marker_prim = marker.get_output_prims()["prims"][0] 
    markers.append(marker)
    marker_prims.append(marker_prim)
    
    # Apply initial texture to marker (will be replaced each iteration)
    with marker:       
        initial_texture = random.choice(tag_textures)
        mat = rep.create.material_omnipbr(
            diffuse_texture=initial_texture,
            emissive_texture=initial_texture,
            emissive_intensity=40.0
        )
        rep.modify.material(mat) 
    
    floating_labeled_prims.append(marker_prim)

print(f"Created {num_markers} marker pool using event-based randomization. Will sample exactly {num_markers_scene} textures from {num_marker_patterns} available patterns each iteration. Each texture ID has a consistent segmentation color.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SHADOWERS  
# shadowers = config.get("shadowers", [])
# for obj in shadowers:
#     obj_url = obj.get("url", "")
#     label = obj.get("label", "unknown")
#     count = obj.get("count", 1)
#     floating = obj.get("floating", False)
#     scale_min_max = obj.get("scale_min_max", (1, 1))
#     for i in range(count):
#         # Create a prim and add the asset reference
#         # rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
#         #     loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
#         # )

#         shadower_plane = rep.create.plane(
#             # position = rep.distribution.uniform((10,10,1), (10,10,2.5)),
#             position = rep.distribution.uniform((-5.0,-5.0,2.5), (5.0,5.0,2.5)),
#             # scale = rep.distribution.uniform((0.01,0.01,0.01), (0.1,0.1,0.1)),
#             scale = rep.distribution.uniform((10.0,10.0,10.0), (10.0,10.0,10.0)),
#             rotation = rep.distribution.uniform((-0,-0,-180), (0,0,180)), 
#             # rotation = (0,0,0),   
#             name = f"shadower_plane_{i}", 
#             semantics=[("class", label)],
#         )
#         shadower_plane_prim = shadower_plane.get_output_prims()["prims"][0] 
#         # set_transform_attributes(shadower_plane_prim, location=rand_loc, rotation=rand_rot, scale=rand_scale) 

#         if floating:
#             floating_labeled_prims.append(shadower_plane_prim)
#         else:
#             falling_labeled_prims.append(shadower_plane_prim)
# print("Shadowers set up.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# ADD BACKGROUND PLANE 
background_plane = rep.create.plane(
    position = (0,0,-10.0),
    scale = (10,10,1), 
    rotation = (0,0,0),   
    name = "background_plane", 
    semantics=[("class", "background")],
)
background_plane_prim = background_plane.get_output_prims()["prims"][0] 
labeled_prims = floating_labeled_prims + falling_labeled_prims
print("Background plane set up.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# ADD DISTRACTOR OBJECTS 
# TODO: add in distractor objects 
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# ENV UPDATE STEP 
simulation_app.update()
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
if disable_render_products_between_captures:
    object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)
print("Environment update step done.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# RANDOMIZER EVENTS 
plane_textures = [os.path.join(dir_backgrounds, f) for f in os.listdir(dir_backgrounds) if os.path.isfile(os.path.join(dir_backgrounds, f))] 

# Background plane randomization
with rep.trigger.on_custom_event(event_name="randomize_plane_texture"): 
    with background_plane:       
        mat = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(plane_textures),
            roughness_texture=rep.distribution.choice(rep.example.TEXTURES),
            metallic_texture=rep.distribution.choice(rep.example.TEXTURES),
            emissive_texture=rep.distribution.choice(rep.example.TEXTURES),
            emissive_intensity=0.0, 
        )    
        rep.modify.material(mat) 

# Lighting randomization
with rep.trigger.on_custom_event(event_name="randomize_lighting"):
    with distant_light:
        rep.modify.pose(
            rotation=rep.distribution.uniform((-30,-30,0), (30,30,0))
        )
        rep.modify.attribute("exposure", rep.distribution.uniform(10, 16)) 

# Initial setup - only for background and lighting
rep.utils.send_og_event(event_name="randomize_plane_texture") 
rep.utils.send_og_event(event_name="randomize_lighting") 

print("Randomizer events set up using simplified approach to avoid hanging (markers handled in simulation loop).")

# set up writer 
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="./output_test", rgb=True, semantic_segmentation=True)
# Attach the actual render product(s)
writer.attach([rp_cam])


#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SDG SETUP 
num_frames = config.get("num_frames", 100)
# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)
# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)
# Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
for _ in range(5):
    simulation_app.update()
# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()
# Store the wall start time for stats
wall_time_start = time.perf_counter()

rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(rp_cam)
sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
sem_annot.attach(rp_cam)
print("SDG setup done.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SIMULATION LOOP 
# Check if we need to randomize marker selection or just use all markers
num_markers_scene = config.get("num_markers_scene", 8)
num_markers = config.get("num_markers", 16)
no_marker_selection_randomization = (num_markers_scene >= num_markers)

if no_marker_selection_randomization:
    print(f"[SDG] All {num_markers} markers will remain active - using hybrid approach")
else:
    print(f"[SDG] Will hide {num_markers - num_markers_scene} markers each frame using hybrid approach")

for i in range(num_frames):
    
    # Memory monitoring every 50 frames
    if i % 50 == 0:
        print_memory_status(i)
    else:
        print(f"[SDG] Processing frame {i}/{num_frames}")
    
    # Determine active markers for this frame
    if not no_marker_selection_randomization:
        # Randomly select which markers to show
        active_marker_indices = random.sample(range(num_markers), num_markers_scene)
    else:
        # All markers are active
        active_marker_indices = list(range(num_markers))
    
    # Store texture paths used for each marker this frame
    frame_marker_textures = {}
    
    # Lighting randomization (less frequent for performance) - use events
    if i % 5 == 0:
        rep.utils.send_og_event(event_name="randomize_lighting")

    # Main randomization every frame - hybrid approach
    if i % 1 == 0:
        # Background texture randomization - use events
        rep.utils.send_og_event(event_name="randomize_plane_texture")
        
        # Marker randomization - direct approach to avoid hanging
        for marker_idx in range(num_markers):
            if marker_idx in active_marker_indices:
                # Active marker - assign texture based on marker index to ensure correspondence
                # tag0 gets tag36_11_00000.png, tag1 gets tag36_11_00001.png, etc.
                if marker_idx < len(tag_textures):
                    # Use texture that corresponds to the marker index
                    texture_filename = f"tag36_11_{marker_idx:05d}.png"
                    texture_path = None
                    # Find the matching texture file
                    for tex_path in tag_textures:
                        if texture_filename in tex_path:
                            texture_path = tex_path
                            break
                    
                    # Fallback to random if specific texture not found
                    if texture_path is None:
                        texture_path = random.choice(tag_textures)
                else:
                    # If we have more markers than textures, cycle through them
                    texture_idx = marker_idx % len(tag_textures)
                    texture_filename = f"tag36_11_{texture_idx:05d}.png"
                    texture_path = None
                    for tex_path in tag_textures:
                        if texture_filename in tex_path:
                            texture_path = tex_path
                            break
                    if texture_path is None:
                        texture_path = tag_textures[texture_idx]
                
                frame_marker_textures[marker_idx] = texture_path  # Store the actual texture used
                with markers[marker_idx]:
                    # Set material
                    mat = rep.create.material_omnipbr(
                        diffuse_texture=texture_path,
                        emissive_texture=texture_path,
                        emissive_intensity=40.0
                    )
                    rep.modify.material(mat)
                    # Set pose using individual rep calls for reliability
                    rep.modify.pose_camera_relative(
                        camera=cam,
                        render_product=rp_cam,
                        distance=random.uniform(config["marker_distance_range"][0], config["marker_distance_range"][1]),
                        horizontal_location=random.uniform(config["marker_horizontal_range"][0], config["marker_horizontal_range"][1]),
                        vertical_location=random.uniform(config["marker_vertical_range"][0], config["marker_vertical_range"][1]),
                    )
                    rep.modify.pose(
                        rotation=(
                            random.uniform(-80, 80),
                            random.uniform(-80, 80), 
                            random.uniform(-180, 180)
                        )
                    )
            else:
                # Inactive marker - move far away
                with markers[marker_idx]:
                    rep.modify.pose(position=(1000, 1000, 1000))

    # Update the app to apply randomization 
    rep.orchestrator.step(delta_time=0.0, rt_subframes=3, pause_timeline=False) 

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    if i % 1 == 0:
        capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=128, apply_blur=False) 
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Collect transform data
    cam_tf = get_world_transform_xform_as_np_tf(cam_prim)
    plane_tf = get_world_transform_xform_as_np_tf(background_plane_prim)
    light_tf = get_world_transform_xform_as_np_tf(distant_light_prim) 
    
    # Collect data for active markers
    markers_data = {}
    marker_ids = []
    current_frame_marker_colors = {}
    
    for marker_idx in active_marker_indices:
        if marker_idx < len(markers):
            marker_prim = marker_prims[marker_idx]
            marker_tf = get_world_transform_xform_as_np_tf(marker_prim)
            
            # Use the actual texture that was applied to this marker
            if marker_idx in frame_marker_textures:
                current_texture_path = frame_marker_textures[marker_idx]
            else:
                # Fallback if something went wrong
                current_texture_path = random.choice(tag_textures)
            
            marker_id = extract_marker_id_from_texture_path(current_texture_path)
            marker_ids.append(marker_id)
            
            markers_data[f"tag{marker_idx}"] = {
                "transform": marker_tf.tolist(),
                "marker_index": marker_idx,
                "texture_id": marker_id,
                "texture_path": current_texture_path,
                "is_active": True
            }
            
            # Map marker label to color for segmentation
            label = f"tag{marker_idx}"
            current_frame_marker_colors[label] = marker_colors_by_texture_id[marker_id]
    
    # Track inactive markers
    for marker_idx in range(num_markers):
        if marker_idx not in active_marker_indices:
            markers_data[f"tag{marker_idx}"] = {
                "marker_index": marker_idx,
                "is_active": False
            }

    pose_data = {
        "cam": cam_tf.tolist(), 
        "markers": markers_data,
        "plane": plane_tf.tolist(), 
        "light": light_tf.tolist(), 
    } 

    # Save data
    write_rgb_data(rgb_annot.get_data(), f"{OUT_DIR}/rgb/rgb_{i}")
    write_sem_data(sem_annot.get_data(), f"{OUT_DIR}/seg/seg_{i}", current_frame_marker_colors)
    write_pose_data(pose_data, f"{OUT_DIR}/pose/pose_{i}") 

    metadata = {
        "light": {
            "exposure": distant_light_lighting_prim.GetAttribute("inputs:exposure").Get(), 
            "color": serialize_vec3f(distant_light_lighting_prim.GetAttribute("inputs:color").Get()), 
        },
        "markers": markers_data,
        "marker_ids": marker_ids,
        "num_active_markers": len(active_marker_indices),
        "total_markers": len(markers),
    } 

    write_metadata(metadata, f"{OUT_DIR}/metadata/metadata_{i}")

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run simulation between captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()
    
    # Memory management - clear caches periodically
    if i % 100 == 0 and i > 0:
        print(f"[SDG] Memory cleanup at frame {i}")
        gc.collect()
        
    if i % 1000 == 0 and i > 0:
        print(f"[SDG] Deep memory cleanup at frame {i}")
        simulation_app.update()
        rep.orchestrator.step(delta_time=0.0, rt_subframes=1, pause_timeline=False)
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# CLEANUP 

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration
num_captures = num_frames * num_cameras
avg_capture_fps = num_captures / wall_duration
print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

# Unsubscribe the physics overlap checks and stop the timeline
# physx_sub.unsubscribe()
# physx_sub = None
simulation_app.update()
timeline.stop()

simulation_app.close()
#------------------------------------------------------------------------------------------------------------------------------------------------------#



