# ~/isaacsim/python.sh synthetic_data_generation/marker_obj_sdg_lights.py 
# ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh data_generation/single_marker_data_generation.py 


# DESCRIPTION: 
# randomize: marker pose, background plane image, lighting direction

# IMPORTS 
import argparse
import json
import os

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
    OUT_DIR = os.path.join(os.getcwd(), "output", "sdg_markers_" + timestr)
    dir_textures = "/home/anegi/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/sdg_tag" 
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
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focalLength": 24.0,
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
    "labeled_assets_and_properties": [
        # {
        #     "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
        #     "label": "pudding_box",
        #     "count": 5,
        #     "floating": True,
        #     "scale_min_max": (0.85, 1.25),
        # },
        # {
        #     "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        #     "label": "mustard_bottle",
        #     "count": 7,
        #     "floating": True,
        #     "scale_min_max": (0.85, 1.25),
        # },
        {
            # "url": "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/Shapes/plane.usd", 
            "label": "tag0", 
            "count": 1, 
            "floating": True, 
            "scale_min_max": (0.1, 0.1), # default plane is 100cm x 100cm, 0.1 scale makes this 10cm x 10cm 
        }, 
    ],
    "shadowers": [
            # plane object 
            {
                "url": "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/Shapes/plane.usd", # FIXME: update to 4.5.0 
                "label": "shadower_plane",
                "count": 1, 
                "floating": True, 
                "scale_min_max": (0.01, 0.1),  
            },
        ], 
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 0,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.015, 0.15),
    "mesh_distractors_num": 0, 
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
def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)

# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

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

# Util function to save rgb annotator data
def write_rgb_data(rgb_data, file_path):
    rgb_img = Image.fromarray(rgb_data, "RGBA")
    rgb_img.save(file_path + ".png")

# Util function to save semantic segmentation annotator data
def write_sem_data(sem_data, file_path):
    id_to_labels = sem_data["info"]["idToLabels"]
    with open(file_path + ".json", "w") as f:
        json.dump(id_to_labels, f)
    sem_image_data = np.frombuffer(sem_data["data"], dtype=np.uint8).reshape(*sem_data["data"].shape, -1)
    sem_img = Image.fromarray(sem_image_data, "RGBA")
    sem_img.save(file_path + ".png")

def write_pose_data(pose_data, file_path):
    with open(file_path + ".json", "w") as f:
        json.dump(pose_data, f) 

def write_metadata(metadata, file_path): 
    with open(file_path + ".json", "w") as f:
        json.dump(metadata, f) 

def serialize_vec3f(vec3f):
    # Convert Gf.Vec3f to a list or dictionary
    return [vec3f[0], vec3f[1], vec3f[2]]

# Update the app until a given simulation duration has passed (simulate the world between captures)
def run_simulation_loop(duration):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )

def quatf_to_eul(quatf): 
    qw = quatf.real 
    qx, qy, qz = np.array(quatf.imaginary) 
    a,b,c = R.from_quat([qx,qy,qz,qw]).as_euler('xyz',degrees=True) 
    return a,b,c 

def get_random_pose_on_hemisphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = abs(radius * np.sin(theta) * np.cos(phi))

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

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
        remove_all_semantics(prim)
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
rp_cam = rep.create.render_product(cam, (640, 480)) 
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
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []
for obj in labeled_assets_and_properties:
    obj_url = obj.get("url", "")
    label = obj.get("label", "unknown")
    count = obj.get("count", 1)
    floating = obj.get("floating", False)
    scale_min_max = obj.get("scale_min_max", (1, 1))
    for i in range(count):
        # Create a prim and add the asset reference
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
        )

        tag = rep.create.plane(
            position = (0,0,-1.0),
            scale = (0.1,0.1,0.1), 
            # rotation = rand_rot,  
            rotation = (0,0,0),   
            name = "tag0", 
            semantics=[("class", label)],
        )
        tag_prim = tag.get_output_prims()["prims"][0] 
        # set_transform_attributes(tag_prim, location=rand_loc, rotation=rand_rot, scale=rand_scale) 
        # add_colliders(tag_prim)
        # add_rigid_body_dynamics(tag_prim, disable_gravity=floating)

        with tag:       
            # tag_texture = rep.distribution.choice(tag_textures)
            selected_texture_path = random.choice(tag_textures)  # This is the actual file path (a string)
            texture_path = selected_texture_path
            tag_texture = rep.distribution.sequence([selected_texture_path])
            mat = rep.create.material_omnipbr(
                # diffuse_texture="/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/tag36h11_0.png",
                # diffuse_texture=tag_textures[0], 
                # diffuse_texture = rep.random.choice(tag_textures), 
                # diffuse_texture = tag_texture, 
                diffuse_texture=rep.distribution.sequence([selected_texture_path]),
                # roughness_texture=rep.distribution.choice(rep.example.TEXTURES),
                # metallic_texture=rep.distribution.choice(rep.example.TEXTURES),
                # emissive_texture=rep.distribution.choice(rep.example.TEXTURES),
                # emissive_intensity=rep.distribution.uniform(0, 1000),
                # emissive_texture=tag_textures[0], 
                # emissive_texture=rep.random.choice(tag_textures),
                # emissive_texture= tag_texture,  
                emissive_texture=rep.distribution.sequence([selected_texture_path]),
                emissive_intensity=40.0, 
            )    
            rep.modify.material(mat) 
        if floating:
            floating_labeled_prims.append(tag_prim)
        else:
            falling_labeled_prims.append(tag_prim)
print("Markers set up.")
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
rep.utils.send_og_event(event_name="randomize_plane_texture") 

with rep.trigger.on_custom_event(event_name="randomize_marker_pose_cam_space"):
    with tag: 
        rep.modify.pose_camera_relative(
            camera=cam, #NOTE: assume single camera 
            render_product=rp_cam,
            distance=rep.distribution.uniform(0.1, 1.2), 
            horizontal_location=rep.distribution.uniform(-1.0, 1.0),
            vertical_location=rep.distribution.uniform(-1.0, 1.0),
            # distance=rep.distribution.uniform(0.010, 5.0), # NOTE: this does not work 
            # horizontal_location=rep.distribution.uniform(0,0),
            # vertical_location=rep.distribution.uniform(0,0),
        )
        rep.modify.pose(
            # rotation=rep.distribution.uniform((-180,-180,-180), (180,180,180)), 
            rotation=rep.distribution.uniform((-80,-80,-180), (80,80,180)), # REDUCED ANGULAR RANGE 
            # rotation=(0,0,0),   
            # position=(0,0,-0.5), 
        )
rep.utils.send_og_event(event_name="randomize_marker_pose_cam_space") 

with rep.trigger.on_custom_event(event_name="randomize_lighting"):
    
    # location, orientation = get_random_pose_on_hemisphere(origin=(0,0,0), radius=1.0, camera_forward_axis=(0,0,-1))
    # a,b,c = quatf_to_eul(orientation) 

    with distant_light:
        rep.modify.pose(
            rotation=rep.distribution.uniform((-30,-30,0), (30,30,0)), # NOTE: believe that this is not perfect but workable, reduced angular range 
            # rotation=(a,b,c),  
        )
        rep.modify.attribute("exposure", rep.distribution.uniform(10, 16)) 
        # rep.modify.attribute("color", rep.distribution.uniform((0, 0, 0), (1, 1, 1)))  
        # rep.modify.attribute("color_temperature", rep.distribution.uniform(2500, 10000))  

rep.utils.send_og_event(event_name="randomize_lighting") 

tag_textures_sequence = random.choices(tag_textures, k=config.get("num_frames", 1_000_000)) 
with rep.trigger.on_custom_event(event_name="randomize_tag_texture"): 
    mat = rep.create.material_omnipbr(
        # diffuse_texture="/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/tag36h11_0.png",
        # diffuse_texture=tag_textures[0], 
        # diffuse_texture = rep.random.choice(tag_textures), 
        # diffuse_texture = tag_texture, 
        diffuse_texture=rep.distribution.sequence(tag_textures_sequence),
        # diffuse_texture=selected_texture_path,
        # roughness_texture=rep.distribution.choice(rep.example.TEXTURES),
        # metallic_texture=rep.distribution.choice(rep.example.TEXTURES),
        # emissive_texture=rep.distribution.choice(rep.example.TEXTURES),
        # emissive_intensity=rep.distribution.uniform(0, 1000),
        # emissive_texture=tag_textures[0], 
        # emissive_texture=rep.random.choice(tag_textures),
        # emissive_texture= tag_texture,  
        emissive_texture=rep.distribution.sequence(tag_textures_sequence),
        # emissive_texture=selected_texture_path,
        emissive_intensity=40.0, 
    )    
    with tag: 
        rep.modify.material(mat) 
rep.utils.send_og_event(event_name="randomize_tag_texture") 

# with rep.trigger.on_custom_event(event_name="randomize_shadower_pose"):   
#     with shadower_plane:
#         rep.modify.pose(
#             # position=rep.distribution.uniform((10,10,1),(10,10,2.5)),
#             position=rep.distribution.uniform((-10.0,-10.0,2.5),(10.0,10.0,2.5)), 
#             rotation=rep.distribution.uniform((-0,-0,-180), (0,0,180)), 
#             scale=rep.distribution.uniform((5.0,5.0,5.0), (10.0,10.0,10.0)), 
#         )
# rep.utils.send_og_event(event_name="randomize_shadower_pose")

print("Randomizer events set up.")

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
sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True})
sem_annot.attach(rp_cam)
print("SDG setup done.")
#------------------------------------------------------------------------------------------------------------------------------------------------------#

# SIMULATION LOOP 
for i in range(num_frames):

    if i % 5 == 0: # NOTE: reduce randomization frequency to speed up compute 
        print(f"Randomize lighting") 
        rep.utils.send_og_event(event_name="randomize_lighting") 

    if i % 1 == 0: 
        print(f"\t Randomizing plane texture") 
        rep.utils.send_og_event(event_name="randomize_plane_texture") 
        
        print(f"Randomize marker pose")
        rep.utils.send_og_event(event_name="randomize_marker_pose_cam_space") 

        # print(f"Randomize shadower pose")
        # rep.utils.send_og_event(event_name="randomize_shadower_pose") 

    if i % 1 == 0: # NOTE: reduce randomization frequency to speed up compute 
        print(f"Randomize tag texture") 
        rep.utils.send_og_event(event_name="randomize_tag_texture") 

    # update the app to apply the randomization 
    rep.orchestrator.step(delta_time=0.0, rt_subframes=3, pause_timeline=False) # NOTE: reducing rt_subframes from 5 for speed 

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if i % 1 == 0:
        # capture_with_motion_blur_and_pathtracing(duration=0.025, num_samples=8, spp=128)
        capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=128, apply_blur=False) 
        # rep.orchestrator.step(delta_time=0.0, rt_subframes=1, pause_timeline=False)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    cam_tf = get_world_transform_xform_as_np_tf(cam_prim)
    tag_tf = get_world_transform_xform_as_np_tf(tag_prim)
    plane_tf = get_world_transform_xform_as_np_tf(background_plane_prim)
    light_tf = get_world_transform_xform_as_np_tf(distant_light_prim) 
    # shadower_tf = get_world_transform_xform_as_np_tf(shadower_plane_prim) 

    pose_data = {
        "cam": cam_tf.tolist(), 
        "tag": tag_tf.tolist(), 
        "plane": plane_tf.tolist(), 
        "light": light_tf.tolist(), 
    } 
    write_rgb_data(rgb_annot.get_data(), f"{OUT_DIR}/rgb/rgb_{i}")
    write_sem_data(sem_annot.get_data(), f"{OUT_DIR}/seg/seg_{i}")
    write_pose_data(pose_data, f"{OUT_DIR}/pose/pose_{i}") 

    # parse number from the path 
    match = re.search(r"_(\d+)\.png$", tag_textures_sequence[i+1]) # off by one because randomizer runs once at initialization 
    if match:
        tag_id = int(match.group(1))

    metadata = {
        "light": {
            "exposure": distant_light_lighting_prim.GetAttribute("inputs:exposure").Get(), 
            "color": serialize_vec3f(distant_light_lighting_prim.GetAttribute("inputs:color").Get()), 
        },
        "tag_id": tag_id
    } 

    write_metadata(metadata, f"{OUT_DIR}/metadata/metadata_{i}")

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()
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



