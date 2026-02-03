import torch

# we set the test camera matrix here

"""
# for hd_full_body
render_cam-world_view_transform:tensor([[    -0.13,     -0.06,      0.99,     -0.00],
    [    -0.00,      1.00,      0.06,      0.00],
    [    -0.99,      0.01,     -0.13,     -0.00],
    [     0.37,     -0.03,      1.44,      1.00]], device='cuda:0')
2025-07-16 11:50:48.564 | INFO     | renderer.gaussian_renderer:_render_impl:184 - render_cam-full_proj_transform:tensor([[    -0.32,     -0.15,      0.99,      0.99],
        [    -0.00,      2.41,      0.06,      0.06],
        [    -2.39,      0.02,     -0.13,     -0.13],
        [     0.90,     -0.08,      1.42,      1.44]], device='cuda:0')
    
    # 
    
# for hair
render_cam-world_view_transform:tensor([[ 0.98,  0.06,  0.19,  0.00],
    [ 0.00,  0.96, -0.30,  0.00],
    [-0.19,  0.29,  0.94,  0.00],
    [-1.58, -2.38,  5.32,  1.00]], device='cuda:0')
render_cam-full_proj_transform:tensor([[ 2.37,  0.14,  0.19,  0.19],
    [ 0.00,  2.31, -0.30, -0.30],
    [-0.47,  0.70,  0.94,  0.94],
    [-3.80, -5.75,  5.31,  5.32]], device='cuda:0')

"""

cams={
}
"""
for garden
render_cam-world_view_transform:tensor([[     0.64,     -0.38,      0.67,     -0.00],
    [    -0.00,      0.87,      0.49,      0.00],
    [    -0.77,     -0.32,      0.56,     -0.00],
    [    -1.37,     -3.61,     11.44,      1.00]], device='cuda:0')
render_cam-full_proj_transform:tensor([[     1.55,     -0.91,      0.67,      0.67],
    [    -0.00,      2.10,      0.49,      0.49],
    [    -1.85,     -0.76,      0.56,      0.56],
    [    -3.30,     -8.71,     11.45,     11.44]], device='cuda:0')
"""
garden_360_cam={
    'world_view_transform':torch.tensor([[     0.99,      0.02,      0.15,      0.00],
        [    -0.00,      0.99,     -0.12,      0.00],
        [    -0.15,      0.12,      0.98,      0.00],
        [    -1.35,     -3.12,     14.17,      1.00]], device='cuda:0'),
    'full_proj_transform':torch.tensor([[     2.39,      0.04,      0.15,      0.15],
        [    -0.00,      2.40,     -0.12,     -0.12],
        [    -0.37,      0.28,      0.98,      0.98],
        [    -3.26,     -7.54,     14.18,     14.17]], device='cuda:0'),
    'camera_center':torch.tensor([ -2.62,   3.61, -15.40], device='cuda:0')
}
"""
template={
    'world_view_transform':torch.,
    'full_proj_transform':torch.,
    'camera_center':torch.
}
"""

hair_1_cam={
    'world_view_transform':torch.tensor([[     0.86,      0.09,     -0.50,      0.00],
        [     0.00,      0.98,      0.18,      0.00],
        [     0.51,     -0.16,      0.84,      0.00],
        [    -2.08,     -1.86,      6.47,      1.00]], device='cuda:0'),
    'full_proj_transform':torch.tensor([[     2.07,      0.22,     -0.51,     -0.50],
        [     0.00,      2.37,      0.18,      0.18],
        [     1.24,     -0.38,      0.85,      0.84],
        [    -5.02,     -4.49,      6.47,      6.47]], device='cuda:0'),
    'camera_center':torch.tensor([ 5.22,  0.66, -4.69], device='cuda:0')
}
hd_full_body_cam={
    'world_view_transform':torch.tensor([[    -0.26,      0.03,      0.97,      0.00],
        [    -0.00,      1.00,     -0.03,      0.00],
        [    -0.97,     -0.01,     -0.26,     -0.00],
        [     0.31,     -0.02,      0.52,      1.00]], device='cuda:0'),
    'full_proj_transform':torch.tensor([[    -0.63,      0.06,      0.97,      0.97],
        [    -0.00,      2.41,     -0.03,     -0.03],
        [    -2.33,     -0.02,     -0.26,     -0.26],
        [     0.76,     -0.05,      0.50,      0.52]], device='cuda:0'),
    'camera_center':torch.tensor([-0.42,  0.03,  0.44], device='cuda:0')
}

hd_lower_body_cam={
    'world_view_transform':torch.tensor([[     0.27,     -0.07,      0.96,      0.00],
        [    -0.00,      1.00,      0.07,      0.00],
        [    -0.96,     -0.02,      0.26,     -0.00],
        [     0.29,     -0.14,      0.74,      1.00]], device='cuda:0'),
    'full_proj_transform':torch.tensor([[     0.64,     -0.16,      0.96,      0.96],
        [    -0.00,      2.41,      0.07,      0.07],
        [    -2.33,     -0.04,      0.27,      0.26],
        [     0.71,     -0.33,      0.73,      0.74]], device='cuda:0'),
    'camera_center':torch.tensor([-0.80,  0.08,  0.08], device='cuda:0')
}

hd_full_leg_cam={
    'world_view_transform':torch.tensor([[     0.72,      0.69,      0.00,      0.00],
        [    -0.00,      0.00,     -1.00,     -0.00],
        [    -0.69,      0.72,      0.00,      0.00],
        [     0.29,     -0.32,      0.73,      1.00]], device='cuda:0'),
    'full_proj_transform':torch.tensor([[     1.74,      1.68,      0.00,      0.00],
        [    -0.00,      0.00,     -1.00,     -1.00],
        [    -1.68,      1.74,      0.00,      0.00],
        [     0.69,     -0.76,      0.72,      0.73]], device='cuda:0'),
    'camera_center':torch.tensor([0.01, 0.73, 0.43], device='cuda:0')
}

# cams['360_garden']=garden_360_cam
# cams['hair_1']=hair_1_cam
# cams['hd_full_body']=hd_full_body_cam
# cams['hd_lower_body']=hd_lower_body_cam
# cams['hd_full_leg']=hd_full_leg_cam