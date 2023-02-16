import os
from typing import Dict
import numpy as np

from ..latents.utils import figure_out_overlap, rgb2gray, get_array_from_png
from ..latents.textures import apply_texture
from ..latents.lights import get_scene_lights

import kubric as kb
from kubric import core
from kubric.simulator import PyBullet
from kubric.renderer import Blender

import shutil
import gc


def render_scene(config: Dict):
    """
    This function takes a config of a single image and creates the scene. Config is a Dict
    with the following keys:

    Keys:
        seed: seed for the random number generator
        resolution: [height, width]
        spawn_region: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        sun_position: [x, y, z]
        camera_position: [x, y, z]
        camera_look_at: [x, y, z]
        camera_focal_length: focal length of the camera
        camera_sensor_width: sensor width of the camera
        floor_scale: [x, y, z]
        floor_position: [x, y, z]
        bg_texture: dictionary with the background texture
        bg_material: dictionary with the background material
        ambient_illumination: ambient illumination of the scene
        num_objects: number of objects in the scene
        object_shapes: array of strings for KuBasic object shapes in the scene
        object_scales: array of scales for the objects in the scene
        object_angles_of_rotation: array of angles of rotation for the objects in the scene
        object_axes_of_rotation: array of axes of rotation for the objects in the scene
        object_quaternions: array of quaternions for the objects in the scene
        object_textures: array of dictionaries with the textures for the objects in the scene
        object_materials: array of dictionaries with the materials for the objects in the scene
        scene_hash: hash of the scene
        
    Returns:
        image: A numpy array representing the image, in 8bit space: [0, 255] as np.unit8
        config: The config of the scene, same as config input but with the object positions added
    """

    rng = np.random.RandomState(seed=config["seed"])
    scene = core.scene.Scene(resolution=config["resolution"])

    scratch_dir = f"./scratch_dir/{config['seed']}"
    os.makedirs(scratch_dir, exist_ok=True)

    sim = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir)
    kubasic = kb.AssetSource.from_manifest(
        "gs://kubric-public/assets/KuBasic/KuBasic.json"
    )

    # Lights
    scene.add(get_scene_lights(position=config["sun_position"]))
    scene.ambient_illumination = kb.Color(
        *3*(config["ambient_illumination"],)
    )

    # Camera
    scene.camera = kb.PerspectiveCamera(
        focal_length=config["camera_focal_length"], 
        sensor_width=config["camera_sensor_width"]
    )
    scene.camera.position = config["camera_position"]
    scene.camera.look_at(config["camera_look_at"])

    # Floor / Background
    floor_material = kb.PrincipledBSDFMaterial(
        **config["bg_material"]
    )

    floor = kb.Cube(
        name=f"floor_{config['seed']}",
        material=floor_material,
        scale=config["floor_scale"], 
        position=config["floor_position"],
        static=True, 
        background=True
    )

    scene += floor
    
    apply_texture(
        obj_name=f"floor_{config['seed']}", 
        material_name=f"floor_material_{config['seed']}", 
        texture=config["bg_texture"]
    )

    # Add random objects
    positions = []

    for i in range(config["num_objects"]):
        # create the object
        obj_name = f"obj{i}_{config['seed']}"
        obj = kubasic.create(
            asset_id=config["object_shapes"][i], 
            scale=config["object_scales"][i], 
            name=obj_name
        )

        # set the object's material
        obj.material = kb.PrincipledBSDFMaterial(
            **config["object_materials"][i]
        )

        # add the object to the scene
        scene += obj
        
        # add texture to the object
        apply_texture(
            obj_name=obj_name, 
            material_name=f"{obj_name}_material", 
            texture=config["object_textures"][i]
        )

        # apply 3D rotation
        obj.quaternion = config["object_quaternions"][i]
        
        # figure out scene placement
        figure_out_overlap(
            asset=obj, 
            simulator=sim, 
            spawn_region=config["spawn_region"], 
            rng=rng
        )
        
        # get the object's metadata
        positions.append(obj.position)

    config["object_positions"] = positions
    # run the simulation
    sim.run()
    # renderer.save_state(scratch_dir + "/scene.blend")

    # get scene and complete metadata
    frame = renderer.render_still(
        return_layers=[
            "rgba", 
            "segmentation", 
            "object_coordinates", 
            "normal", 
            "depth"
        ]
    )
    
    # get grayscale scene from rgba
    frame["grayscale"] = frame.pop("rgba")
    frame["grayscale"] = rgb2gray(frame["grayscale"]).astype(np.uint8)

    # process segmentation
    frame['segmentation'] = frame['segmentation'].astype(np.uint8)
    frame['segmentation'] -= np.min(frame['segmentation'])

    # process object coordinates
    frame["object_coordinates"] = get_array_from_png(
        frame["object_coordinates"], 
        scratch_dir + "/object_coordinates.png"
    )

    # process normals
    frame["normal"] = get_array_from_png(
        frame["normal"], 
        scratch_dir + "/normal.png"
    )

    # process depth
    min_value = np.min(frame["depth"])
    max_value = np.max(frame["depth"])
    
    frame["depth"] = (frame["depth"] - min_value) * 65535 / (max_value - min_value)
    frame["depth"] = frame["depth"].astype(np.uint16)

    frame["depth"] = get_array_from_png(
        frame["depth"], 
        scratch_dir + "/depth.png"
    )

    config["depth_scaling"] = {
        "min_depth": min_value.item(), 
        "max_depth": max_value.item()
    }

    kb.done()

    # clean up
    shutil.rmtree(scratch_dir)
    gc.collect()

    return frame, config
    
