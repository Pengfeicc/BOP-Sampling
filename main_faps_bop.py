import blenderproc as bproc
import argparse
import os
import numpy as np
from pathlib import Path
import random
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="exp/physics_positioning/plane.obj", help="Path to the bop datasets parent directory")
parser.add_argument('bop_parent_path', nargs='?', default="exp/physics_positioning/bop_path", help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', default="lm", help="Main BOP dataset")
parser.add_argument('image_dir', nargs='?', default="exp/physics_positioning/images", help="Path to a folder with .jpg textures to be used in the sampling process")
parser.add_argument('output_dir', default="exp/physics_positioning/output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()
# load the objects into the scene
plane = bproc.loader.load_obj(args.scene)[0]
plane.set_rotation_euler([np.pi, 0, 0])
plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

materials = bproc.material.collect_all()

# Find the material of the ground object
ground_material = bproc.filter.one_by_attr(materials, "name", "Material.001")
# Set its displacement based on its base color texture
ground_material.set_displacement_from_principled_shader_value("Base Color", multiply_factor=1.5)

# Collect all jpg images in the specified directory
images = list(Path(args.image_dir).absolute().rglob("material_manipulation_sample_texture2.jpg"))
for mat in materials:
    # Load one random image
    image = bpy.data.images.load(filepath=str(random.choice(images)))
    # Set it as base color of the current material
    mat.set_principled_shader_value("Base Color", image)

# load a random sample of bop objects into the scene
sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'),
                                  mm2m = True,
                                  sample_objects = True,
                                  num_of_objs_to_sample = 10)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(sampled_bop_objs):
    obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    obj.set_shading_mode('auto')
        
    mat = obj.get_materials()[0]
    if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
        grey_col = np.random.uniform(0.1, 0.9)   
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

    obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, collision_margin=0.0005)
    obj.hide(False)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample object poses and check collisions 
bproc.object.sample_poses(objects_to_sample = sampled_bop_objs,
                        sample_pose_func = sample_pose_func, 
                        max_tries = 1000)
        
# Physics Positioning
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                max_simulation_time=10,
                                                check_object_interval=1,
                                                substeps_per_frame = 20,
                                                solver_iters=25)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)

poi = bproc.object.compute_poi(sampled_bop_objs)
cam_poses = 0
while cam_poses < 10:
    # Sample location
    location = bproc.sampler.shell(center = [0, 0, 0],
                            radius_min = 0.8,
                            radius_max = 1.2,
                            elevation_min = 45,
                            elevation_max = 89)
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    
    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        cam_poses += 1

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# render the whole pipeline
data = bproc.renderer.render()

# Write data in bop format
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                       dataset = 'lm',
                       depths = data["depth"],
                       colors = data["colors"], 
                       color_file_format = "JPEG",
                       ignore_dist_thres = 10)