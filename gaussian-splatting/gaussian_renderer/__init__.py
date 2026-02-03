#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import open3d as o3d
import torch
import math
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import time
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
# from ray_tracer import tracer
from loguru import logger
import numpy as np
from PIL import Image
import os
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    

    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_simple(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, debug=False,**other_args):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    ellapsed_time=0.0
    start = time.perf_counter()
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # check clipping info  DL: precheck the params, only the widget is click we can get the default params
    n_clips=0
    clippers= torch.tensor([1.0, 0.0,0.0, 1], dtype=pc.get_xyz.dtype,device=pc.get_xyz.device)  # (normal, distance)
    rr_clipping=False
    if "clipping_planes" in other_args.keys():
        n_clips=len(other_args['clipping_planes'])
        clippers= torch.zeros(n_clips, 4, device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
        for n in range(n_clips):# fill all clippers
            clipper_n=other_args['clipping_planes'][n]
            clippers[n][0]=clipper_n['normal'][0]
            clippers[n][1]=clipper_n['normal'][1]
            clippers[n][2]=clipper_n['normal'][2]
            clippers[n][3]=clipper_n['d']
    if "rr_clipping" in other_args.keys():
        rr_clipping=other_args['rr_clipping']
    else:
        pass
    
    vizPlane=False
    n_inters=0
    intersections_tensor=torch.tensor([1.0, 0.0,0.0], dtype=pc.get_xyz.dtype,device=pc.get_xyz.device)  # (normal, distance)
    if "vizPlane" in other_args.keys() and other_args['vizPlane']:
        vizPlane=other_args['vizPlane']
        intersections_tensor=other_args['intersections_tensor']
        n_inters=other_args['n_inters']
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    rr_strategy=False
    if "rr_strategy" in other_args.keys():
        rr_strategy=other_args['rr_strategy']
    oenD_gs_strategy=False
    if "oenD_gs_strategy" in other_args.keys():
        oenD_gs_strategy=other_args['oenD_gs_strategy']

    # check evs splitting info
    enable_evs=False
    if "enable_evs" in other_args.keys():
        enable_evs=other_args['enable_evs']
    evs_debug=False
    if "evs_debug" in other_args.keys():
        evs_debug=other_args['evs_debug']


    # Benefit-cost split control parameters
    evs_split_mode = other_args.get('evs_split_mode', 0)    # 0=naive, 1=proxy_control
    evs_cost_mode = other_args.get('evs_cost_mode', 0)      # 0=1-min(Cl,Cr), 1=|Cl-Cr|
    evs_lambda = other_args.get('evs_lambda', 1.0)          # benefit-cost threshold

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=evs_debug,
        rr_clipping=rr_clipping,
        rr_strategy=rr_strategy,
        oenD_gs_strategy=oenD_gs_strategy,
        n_clips=n_clips,
        clipprs=clippers,
        vizPlane=vizPlane,
        n_inters=n_inters,
        intersections_tensor=intersections_tensor,
    )
    
    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rasterizer=GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    convert_shs_python = False
    colors_precomp = None
    if override_color is None:
        if convert_shs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # torch.cuda.synchronize()
    end_time = time.perf_counter()

    ellapsed_time=end_time-start

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
        "ellapsed_time":ellapsed_time
    }
