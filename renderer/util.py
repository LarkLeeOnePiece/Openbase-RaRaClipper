import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

import numpy as np
from skimage.io import imread, imsave
from imageio import v2 as imageio
import cv2
import matplotlib.pyplot as plt
def compute_l1_loss(img_path1, img_path2, resize=True):
    img1 = imread(img_path1).astype(np.float32)
    img2 = imread(img_path2).astype(np.float32)

    if img1.shape != img2.shape:
        if not resize:
            raise ValueError("Image shapes do not match and resize=False")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    l1 = np.mean(np.abs(img1 - img2))
    return l1

def compute_ssim_auto(img_path1, img_path2, resize=True, verbose=False):
    """
    自动检测图像类型并计算 SSIM
    :param img_path1: Ground truth 图像路径
    :param img_path2: 重建图像路径
    :param resize: 如果尺寸不同，是否自动 resize
    :param verbose: 是否打印中间信息
    :return: SSIM 值
    """
    # 读取图像
    img1 = imread(img_path1)
    img2 = imread(img_path2)

    # 如果尺寸不同，自动 resize
    if img1.shape != img2.shape:
        if not resize:
            raise ValueError("Images have different sizes and resize=False")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        if verbose:
            pass

    # 自动灰度化
    if img1.ndim == 3:
        img1 = rgb2gray(img1)
    if img2.ndim == 3:
        img2 = rgb2gray(img2)

    # 自动判断 data_range
    if img1.dtype == np.uint8:
        data_range = 255
    elif img1.dtype in [np.float32, np.float64]:
        data_range = 1.0 if img1.max() <= 1.0 else 255.0
    else:
        raise ValueError(f"Unsupported image dtype: {img1.dtype}")

    if verbose:
            pass

    # 计算 SSIM
    score, _ = ssim(img1, img2, data_range=data_range, full=True)
    return score

def psnr(img_path1, img_path2, resize=True, verbose=False):
    """
    计算两幅图像的 PSNR（dB），返回一个 float。
    
    参数：
      img_path1, img_path2: 图像文件路径
      resize (bool): 若两图尺寸不同，是否将第二张图 resize 到第一张图大小
      verbose (bool): 是否打印中间信息
    
    返回：
      PSNR 值 (float)，如果 mse=0 则返回 math.inf
    """
    # 1. 读取
    img1 = imageio.imread(img_path1)
    img2 = imageio.imread(img_path2)
    
    # 2. resize
    if img1.shape != img2.shape:
        if not resize:
            raise ValueError("Images have different sizes and resize=False")
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
        if verbose:
            pass
    
    # 3. 转灰度（若有三通道）
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # 4. 确定 data_range
    if np.issubdtype(img1.dtype, np.integer):
        data_range = float(np.iinfo(img1.dtype).max)
    elif np.issubdtype(img1.dtype, np.floating):
        # 假设归一到 [0,1] 或 [0,255]
        data_range = 1.0 if img1.max() <= 1.0 else 255.0
    else:
        raise ValueError(f"Unsupported image dtype: {img1.dtype}")
    
    if verbose:
            pass
    
    # 5. 计算 MSE
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # 6. 计算 PSNR
    return 20 * np.log10(data_range / np.sqrt(mse))

# def save_error_map(img_path1, img_path2, save_path="error_map.png", resize=True, mode='L1'):
    pass
#     img1 = imread(img_path1).astype(np.float32)
#     img2 = imread(img_path2).astype(np.float32)

#     if img1.shape != img2.shape:
    pass
#         if not resize:
    pass
#             raise ValueError("Images must match in shape or set resize=True")
#         img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#     # 计算误差图
#     if img1.ndim == 3:
    pass
#         if mode == 'L1':
    pass
#             err = np.mean(np.abs(img1 - img2), axis=-1)
#         elif mode == 'L2':
    pass
#             err = np.sqrt(np.mean((img1 - img2) ** 2, axis=-1))
#     else:
    pass
#         if mode == 'L1':
    pass
#             err = np.abs(img1 - img2)
#         elif mode == 'L2':
    pass
#             err = np.sqrt((img1 - img2) ** 2)

#     # 归一化
#     err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
#     err_boosted = err_norm 
#     plt.imsave(save_path, err_boosted, cmap='hot')

def save_error_map(img_path1, img_path2, save_path="error_map.png", resize=True, mode='L1'):
    img1 = imread(img_path1).astype(np.float32)
    img2 = imread(img_path2).astype(np.float32)

    if img1.shape != img2.shape:
        if not resize:
            raise ValueError("Images must match in shape or set resize=True")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算误差图
    if img1.ndim == 3:
        if mode == 'L1':
            err = np.mean(np.abs(img1 - img2), axis=-1)
        elif mode == 'L2':
            err = np.sqrt(np.mean((img1 - img2) ** 2, axis=-1))
    else:
        if mode == 'L1':
            err = np.abs(img1 - img2)
        elif mode == 'L2':
            err = np.sqrt((img1 - img2) ** 2)

    # 可选归一化（你可以控制是否归一化）
    err_min, err_max = np.min(err), np.max(err)
    norm_err = (err - err_min) / (err_max - err_min + 1e-8)

    # 画图：蓝底、加色条
    plt.figure(figsize=(6, 5))
    im = plt.imshow(norm_err, cmap='jet', vmin=0, vmax=1)
    cbar = plt.colorbar(im)
    cbar.set_label("Error", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
