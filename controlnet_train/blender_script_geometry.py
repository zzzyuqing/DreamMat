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
render.image_settings.color_mode = "RGB"
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
        if obj.type not in {"CAMERA"}:#LIGHT
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
def create_group_view_normal():
    # 创建一个新的节点组
    node_group = bpy.data.node_groups.new('View_Normal', 'CompositorNodeTree')

    # 添加输入和输出节点
    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')

    normal_input = node_group.inputs.new('NodeSocketVector', 'Normal')
    depth_input = node_group.inputs.new('NodeSocketFloat', 'Depth')


    # 添加其他节点
    vn0 = node_group.nodes.new("CompositorNodeNormal")
    vn1 = node_group.nodes.new("CompositorNodeNormal")
    vn2 = node_group.nodes.new("CompositorNodeNormal")
    vn0.label = "0"
    vn1.label = "1"
    vn2.label = "2"

    map0= node_group.nodes.new("CompositorNodeMath")
    map1= node_group.nodes.new("CompositorNodeMath")
    map2= node_group.nodes.new("CompositorNodeMath")
    map0.operation = "MULTIPLY_ADD"
    map1.operation = "MULTIPLY_ADD"
    map2.operation = "MULTIPLY_ADD"
    # only yz need to inverse
    map0.inputs[1].default_value = 0.5
    map0.inputs[2].default_value = 0.5
    map1.inputs[1].default_value = -0.5
    map1.inputs[2].default_value = 0.5
    map2.inputs[1].default_value = 0.5
    map2.inputs[2].default_value = 0.5

    # mask
    mask_lessthan = node_group.nodes.new("CompositorNodeMath")
    mask_inverse = node_group.nodes.new("CompositorNodeMath")
    mask_ma = node_group.nodes.new("CompositorNodeMath")

    mask_lessthan.operation = "LESS_THAN"
    mask_inverse.operation = "MULTIPLY_ADD"
    mask_ma.operation = "MULTIPLY_ADD"
     
    mask_lessthan.inputs[1].default_value = 100.0
    mask_inverse.inputs[1].default_value = -1.0
    mask_inverse.inputs[2].default_value = 1.0 
    mask_ma.inputs[1].default_value = -1.0
     

    # 将所有节点连接起来
    node_group.links.new(input_node.outputs['Normal'],vn0.inputs[0])
    node_group.links.new(input_node.outputs['Normal'],vn1.inputs[0])
    node_group.links.new(input_node.outputs['Normal'],vn2.inputs[0])

    node_group.links.new(vn0.outputs[1], map0.inputs[0])
    node_group.links.new(vn1.outputs[1], map1.inputs[0])
    #node_group.links.new(vn2.outputs[1], map0.inputs[0])

    node_group.links.new(vn2.outputs[1], mask_ma.inputs[0])
    node_group.links.new(input_node.outputs['Depth'], mask_lessthan.inputs[0])
    node_group.links.new(mask_lessthan.outputs[0], mask_inverse.inputs[0])
    node_group.links.new(mask_inverse.outputs[0], mask_ma.inputs[2])
    node_group.links.new(mask_ma.outputs[0], map2.inputs[0])

    #output
    combine = node_group.nodes.new("CompositorNodeCombineXYZ")
    node_group.links.new(map0.outputs[0],combine.inputs[0])
    node_group.links.new(map1.outputs[0],combine.inputs[1])
    node_group.links.new(map2.outputs[0],combine.inputs[2])
    node_group.outputs.new('NodeSocketVector', 'viewNormal')
    node_group.links.new(combine.outputs[0],output_node.inputs[0])

    # 将新的节点组添加到 Compositor 中
    group_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeGroup')
    bpy.context.scene.node_tree.nodes[-1].node_tree = node_group
    return (group_node,vn0,vn1,vn2)

def save_images(object_file: str) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)

    depthdir=os.path.join(args.output_dir,object_uid,"depth")
    normaldir=os.path.join(args.output_dir,object_uid,"normal")

    if not os.path.exists(depthdir): Path(depthdir).mkdir(exist_ok=True, parents=True)
    if not os.path.exists(normaldir): Path(normaldir).mkdir(exist_ok=True, parents=True)

    
    reset_scene()
    # load the object
    load_object(object_file)
    normalize_scene()

    # load env_map
    bpy.context.scene.world.use_nodes = True

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

    default_color_management = bpy.context.scene.view_settings.view_transform# = 'Filmic'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    render_layer = tree.nodes.new(type="CompositorNodeRLayers")
    composite_node = tree.nodes.new(type="CompositorNodeComposite")
    tree.nodes.active = composite_node

    # depth control Nodes
    subtract_node = tree.nodes.new(type="CompositorNodeMath")
    subtract_node.operation = "SUBTRACT"
    subtract_node.inputs[1].default_value = 100.0
    lessthan_node = tree.nodes.new(type="CompositorNodeMath")
    lessthan_node.operation = "LESS_THAN"
    lessthan_node.inputs[1].default_value = 0.0
    tree.links.new(subtract_node.outputs[0], lessthan_node.inputs[0])

    multip_node_0 = tree.nodes.new(type="CompositorNodeMath")
    multip_node_1 = tree.nodes.new(type="CompositorNodeMath")
    divide_node = tree.nodes.new(type="CompositorNodeMath")
    multip_node_0.operation = "MULTIPLY"
    multip_node_1.operation = "MULTIPLY"
    divide_node.operation = "DIVIDE"
    multip_node_1.inputs[1].default_value = 1000.0
    divide_node.inputs[1].default_value = 65535.0
    tree.links.new(multip_node_0.outputs[0], multip_node_1.inputs[0])
    tree.links.new(multip_node_1.outputs[0], divide_node.inputs[0])

    tree.links.new(render_layer.outputs["Depth"], subtract_node.inputs[0])
    tree.links.new(render_layer.outputs["Depth"], multip_node_0.inputs[0])
    tree.links.new(lessthan_node.outputs[0],multip_node_0.inputs[1])
    output_depth_slot = divide_node.outputs[0]
    # output - divide_node.outputs[0]
    (node_group,vn0,vn1,vn2) = create_group_view_normal()
    tree.links.new(render_layer.outputs["Depth"], node_group.inputs['Depth'])
    tree.links.new(render_layer.outputs["Normal"], node_group.inputs['Normal'])

    distances = np.asarray([1.5 for _ in range(args.num_images)])

    azimuths = (np.arange(args.num_images/2)/args.num_images*np.pi*4).astype(np.float32)
    azimuths = np.concatenate((azimuths, azimuths))
    elevations = np.deg2rad(np.asarray([args.elevation] * (args.num_images//2)).astype(np.float32))
    elevations_2 = np.deg2rad(np.asarray([0.0] * (args.num_images//2)).astype(np.float32))
    elevations = np.concatenate((elevations_2,elevations))

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    
    render.image_settings.compression = 0
    for i in range(args.num_images):
        # set camera
        camera = set_camera_location(cam_pts[i])
        # output depth image
        bpy.context.scene.view_settings.view_transform = "Raw"
        render.image_settings.color_mode = "BW"
        render.image_settings.color_depth = "16"
        tree.links.new(output_depth_slot, composite_node.inputs[0])#render_layer.outputs["Normal"]
        render_path = os.path.join(depthdir,f"{i:03d}.png")
        #if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

        #output normal image
        view_matrix= calc_view_matrix(camera)
        vn0.outputs[0].default_value = view_matrix[0]
        vn1.outputs[0].default_value = view_matrix[1]
        vn2.outputs[0].default_value = view_matrix[2]

        render.image_settings.color_mode = "RGB"
        render.image_settings.color_depth = "8"
        tree.links.new(node_group.outputs["viewNormal"], composite_node.inputs[0])
        render_path = os.path.join(normaldir,f"{i:03d}.png")
        #if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)
        

    print("done")

        
if __name__ == "__main__":
    save_images(args.object_path)
