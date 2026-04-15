import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from skimage import measure, transform
import torchvision.utils
import torch
import torch.nn.functional as F
from math import *
import cv2
import math

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    descriptors_a = descriptors_a.to(device=device)
    descriptors_b = descriptors_b.to(device=device)
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy(), sim


def rancac(kp1_all, kp2_all):
    locations_1_to_use = np.array(kp1_all)
    locations_2_to_use = np.array(kp2_all)
    _RESIDUAL_THRESHOLD = 2
    _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                transform.AffineTransform,
                                min_samples=3,
                                residual_threshold=_RESIDUAL_THRESHOLD,
                                max_trials=1000)
    inlier_idxs = np.nonzero(inliers)[0]  # 返回数组第一维中非零元素的索引值
    locations_1_to_use = locations_1_to_use.tolist()
    locations_2_to_use = locations_2_to_use.tolist()
    locations_1_to_use1 = []
    locations_2_to_use1 = []
    for g in inlier_idxs:
        locations_1_to_use1.append(locations_1_to_use[g])
        locations_2_to_use1.append(locations_2_to_use[g])
    return np.array(locations_1_to_use1), np.array(locations_2_to_use1)


def warpPerspectivePoints(src_points, H):
    # normalize H
    if H.shape[0] == 4 and H.shape[1] == 4:
        M = H
        n = src_points.shape[0]
        points_homogeneous = np.hstack((src_points, np.ones((n, 1)), np.ones((n, 1))))

        # 使用变换矩阵 M 将点映射到图像2中
        mapped_points_homogeneous = np.dot(M, points_homogeneous.T).T

        # 将齐次坐标转换回非齐次坐标
        mapped_points_homogeneous /= mapped_points_homogeneous[:, 3][:, np.newaxis]

        # 提取新的二维坐标
        mapped_points = mapped_points_homogeneous[:, :2]
        warpPoints = mapped_points
    else:
        if H.shape[0] == 3 and H.shape[0] == 3:
            H = H
        else:
            H = np.row_stack((H, [0, 0, 1]))
        H /= H[2][2]

        ones = np.ones((src_points.shape[0], 1))
        points = np.append(src_points, ones, axis=1)
        warpPoints = np.dot(H, points.T)
        warpPoints = warpPoints.T / warpPoints.T[:, 2][:, None]
    return warpPoints[:, 0:2]


def image_compose(img0, img1, w1, h1, w2, h2, s):
    if h1 > h2:
        h = h1
        to_image = Image.new('RGB', ((w1 + w2 + s), h), color=(255, 255, 255))  # 创建一个新图
        to_image.paste(img0, (0, 0))
        to_image.paste(img1, (w1 + s, int(h1 / 2) - int(h2 / 2)))
    else:
        h = h2
        to_image = Image.new('RGB', ((w1 + w2 + s), h), color=(255, 255, 255))  # 创建一个新图
        to_image.paste(img0, (0, int(h2 / 2) - int(h1 / 2)))
        to_image.paste(img1, (w1 + s, 0))

    return to_image  # 新图


def draw_match(img1, img2, kp1, kp2, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    s1 = 0
    all_img = np.asarray(
        image_compose(
            Image.fromarray(img1), Image.fromarray(img2),
            w1, h1, w2, h2, s1))
    green = (0, 255, 0)
    try:
        for k in range(len(kp1)):
            if h2 > h1:
                cv2.line(all_img,
                         (int(kp1[k][0]),
                          int(kp1[k][1] + (int(h2 / 2) - int(h1 / 2)))),
                         (int(kp2[k][0]) + w1 + s1,
                          int(kp2[k][1])), green, 1)
                # cv2.circle(all_img, (int(kp1[k][0]), int(kp1[k][1] + (int(h2 / 2) - int(h1 / 2)))), 5, (0, 0, 255), -1)
                # cv2.circle(all_img, (int(kp2[k][0]) + w1 + s1, int(kp2[k][1])), 5, (255, 0, 0), -1)
            else:
                cv2.line(all_img,
                         (int(kp1[k][0]),
                          int(kp1[k][1])),
                         (int(kp2[k][0]) + w1 + s1,
                          int(kp2[k][1]) + int(int(h1 / 2) - int(h2 / 2))), green, 1)
                # cv2.circle(all_img, (int(kp1[k][0]), int(kp1[k][1])), 5, (0, 0, 255), -1)
                # cv2.circle(all_img, (int(kp2[k][0]) + w1 + s1, int(kp2[k][1]) + int(int(h1 / 2) - int(h2 / 2))), 5, (255, 0, 0), -1)
    except:
        pass
    return all_img


def savevis(img1, img2, kp1, kp2, path):
    vis = draw_match(img1, img2, kp1, kp2)
    cv2.imwrite(path, vis)



def c_rmse_NCM_NewKP_thres(vp1_before_h, vp2, h, thres):
    vp1 = warpPerspectivePoints(vp1_before_h, h)
    distances = []
    for (x1, y1), (x2, y2) in zip(vp1, vp2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)
    elements_less_than_3_with_index = [(index, value) for index, value in enumerate(distances) if value < thres]
    indexes_less_than_3 = [index for index, value in elements_less_than_3_with_index]  ##索引
    values_less_than_3 = [value for index, value in elements_less_than_3_with_index]  ##值
    if len(values_less_than_3) > 0:
        average = sum(values_less_than_3) / len(values_less_than_3)
        NCM = len(values_less_than_3)
        new_vp1 = [vp1_before_h[index] for index in indexes_less_than_3]
        new_vp2 = [vp2[index] for index in indexes_less_than_3]
        new_vp1 = np.array(new_vp1)
        new_vp2 = np.array(new_vp2)
    else:
        average = 20  # 如果没有小于3的元素，则平均值为0
        NCM = 0
        new_vp1 = None
        new_vp2 = None
    rmse = average
    return rmse, NCM, new_vp1, new_vp2

def c_rmse(vp1, vp2, h):
    vp1 = warpPerspectivePoints(vp1, h)
    distances = []
    for (x1, y1), (x2, y2) in zip(vp1, vp2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)
    rmse = np.mean(np.array(distances))
    return rmse


def adaptive_image_pyramid(img, min_scale=0.0, max_scale=1, min_size=128, max_size=1536, scale_f=2 ** 0.25,
                               verbose=False):
    B, _, H, W = img.shape

    ## upsample the input to bigger size.
    s = 1.0
    if max(H, W) < max_size:
        s = max_size / max(H, W)
        max_scale = s
        nh, nw = round(H * s), round(W * s)
        # if verbose:  print(f"extracting at highest scale x{s:.02f} = {nw:4d}x{nh:3d}")
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    ## downsample the scale pyramid
    output = []
    scales = []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]

            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            output.append(img)
            scales.append(s)
        s /= scale_f
        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
    return output, scales