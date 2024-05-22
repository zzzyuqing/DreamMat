import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from mathutils import Vector, Matrix
import numpy as np
import bpy
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--env_dir", type=str, required=True)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--device", type=str, default='CUDA')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
render.engine = "CYCLES"
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100
render.use_persistent_data = True
scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device # or "OPENCL"
bpy.context.scene.cycles.tile_size = 8192

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    return camera


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def calc_view_matrix(cam):
    loc,rot = cam.matrix_world.decompose()[0:2]
    rot = rot.to_matrix()
    rot.transpose()
    return rot

def create_material():
    mat = bpy.data.materials.new('RM')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.0
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Specular'].default_value = 1.0
    bsdf.inputs['Transmission'].default_value = 0.0
    bsdf.inputs['Sheen Tint'].default_value = 0.0
    bsdf.inputs['Sheen'].default_value = 0.0
    bsdf.inputs['Clearcoat Roughness'].default_value = 0.0
    bsdf.inputs['Anisotropic'].default_value = 0.0
    return mat
def save_images(object_file: str) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)
    colordir=os.path.join(args.output_dir,object_uid,"color")
    lightdir = os.path.join(args.output_dir,object_uid,"light")
    if not os.path.exists(colordir): Path(colordir).mkdir(exist_ok=True, parents=True)
    if not os.path.exists(lightdir): Path(lightdir).mkdir(exist_ok=True, parents=True)


    reset_scene()
    # load the object
    load_object(object_file)
    # object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    mesh_list = list(scene_meshes())
    default_mat_list =  [mesh.active_material for mesh in mesh_list]
    mat = create_material()

    # load env_map
    bpy.context.scene.world.use_nodes = True

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']

    map_path = args.env_dir

    mat_list = [(0.0,0.0),(0.0,0.5),(0.0,1.0),(1.0,0.0),(1.0,0.5),(1.0,1.0)]
    env_texture_node = world_tree.nodes.new(type="ShaderNodeTexEnvironment")
    world_tree.links.new(env_texture_node.outputs[0], back_node.inputs[0])
    distances = np.asarray([1.5 for _ in range(args.num_images)])

    azimuths = (np.arange(args.num_images/2)/args.num_images*np.pi*4).astype(np.float32)
    azimuths = np.concatenate((azimuths, azimuths))
    elevations = np.deg2rad(np.asarray([args.elevation] * (args.num_images//2)).astype(np.float32))
    elevations_2 = np.deg2rad(np.asarray([0.0] * (args.num_images//2)).astype(np.float32))
    elevations = np.concatenate((elevations_2,elevations))

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    cam_poses = []

    render.image_settings.color_mode = "RGBA"
    for env in range(5):
        env_map = bpy.data.images.load(os.path.abspath(os.path.join(map_path,'map'+str(env+1), 'map'+str(env+1)+".exr")))
        env_texture_node.image = env_map
        for i in range(args.num_images):
            # set camera
            camera = set_camera_location(cam_pts[i])

            for mesh in mesh_list:
                mesh.active_material = mat
            for j in range(len(mat_list)):
                render_path = os.path.join(lightdir,f"{i:03d}_m"+str(mat_list[j][0])+"r"+str(mat_list[j][1])+"_env"+str(env+1)+".png")
                #if os.path.exists(render_path): continue
                mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = mat_list[j][0]
                mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = mat_list[j][1]
                scene.render.filepath = os.path.abspath(render_path)
                bpy.ops.render.render(write_still=True)

            # output render image
            render_path = os.path.join(colordir,f"{i:03d}_color_env"+str(env+1)+".png")
            #if os.path.exists(render_path): continue
            for mesh in mesh_list:
                mesh.active_material = default_mat_list[mesh_list.index(mesh)]
            scene.render.filepath = os.path.abspath(render_path)
            bpy.ops.render.render(write_still=True)

    print("done")

        
if __name__ == "__main__":
    save_images(args.object_path)
