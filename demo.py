import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import random
import torchvision.transforms.functional as TF
import sys
from all_base import *
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.model import REMM
import cv2
from PIL import Image
import torchvision
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
matplotlib.use('Agg')


def save_features2png(features, out_path):
    # 将特征图移到CPU并转换为numpy格式
    features = features.detach().cpu()
    if features.shape[1] == 1:
        features = features.squeeze(1)  # 压缩单通道到 HxW
        grid = features.numpy().transpose((1, 2, 0))  # 转置为 (H, W, C) 格式
        grid = (grid - grid.min()) / (grid.max() - grid.min()) * 255  # 归一化到 0-255
        grid = grid.astype(np.uint8)
        pil_image = Image.fromarray(grid.squeeze(-1))  # 创建 PIL 图像
    else:
        grid = torchvision.utils.make_grid(features[:, :2, :, :], nrow=int(np.sqrt(features.size(0))), normalize=True)
        pil_image = torchvision.transforms.ToPILImage()(grid)
    pil_image.save(out_path)


def load_network(model_name, model_fn):
    if model_name == "REMM":
        checkpoint = torch.load(model_fn)
        model = REMM()
        try:
            weights = checkpoint["model"]
        except:
            weights = checkpoint
        model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return model.eval()


def shift_topk_candidate(desc, topk=1):
    B, K, CG = desc.shape
    desc = desc.reshape(B, K, -1, 16)
    shifts = torch.topk(desc[:, :, 0, :], k=topk, dim=2)[1]
    desc = desc.reshape(B * K, -1, CG)
    shifts = shifts.reshape(B * K, -1)

    desc_update = []
    for shift in shifts.t():
        for d, s in zip(desc, shift):
            desc_update.append(torch.roll(d, shifts=-int(s), dims=-1))  ## reverse shift

    desc_update = torch.stack(desc_update)  ## [topk*B*K, C, G]
    desc_update = desc_update.reshape(topk, B, K, -1, CG).transpose(0, 1)  ## [B*topk*K, C, G]
    desc_update = desc_update.reshape(B, topk * K, -1)
    return desc_update


def shift_ratio_candidate(kpts, desc, ratio=1.0):
    B, K, CG = desc.shape
    desc = desc.reshape(B, K, -1, 16)
    value, _ = torch.max(desc[:, :, 0, :], dim=2)
    ratio_tensor = (desc[:, :, 0, :] / value.unsqueeze(-1))  ## obtain ratio
    ratio_mask = ratio_tensor >= ratio
    kpts_update = []
    desc_update = []
    for _kpts, _desc, _ratio_mask in zip(kpts, desc, ratio_mask):
        kpts_update_iter = []
        desc_update_iter = []
        for k, d, r in zip(_kpts, _desc, _ratio_mask):
            shifts = r.nonzero().reshape(-1)
            for s in shifts:
                kpts_update_iter.append(k)
                desc_update_iter.append(torch.roll(d, shifts=-int(s), dims=-1))

        kpts_update.append(torch.stack(kpts_update_iter))
        desc_update.append(torch.stack(desc_update_iter).reshape(-1, CG))
    return kpts_update[0], desc_update[0]


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression, self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr

    def forward(self, repeatability):
        # repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima * 0
        border_mask[:, :, 10:-10, 10:-10] = 1
        maxima = maxima * border_mask
        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, image_type,
                       scale_f=2 ** 0.25, min_scale=0.0,
                       max_scale=1, min_size=128,
                       max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            # if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img)

            mask = repeatability * 0
            border = 5
            mask[:, :, border:-border, border:-border] = 1
            repeatability = repeatability * mask
            y, x = detector(repeatability)  # nms
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            Q.append(q)
            D.append(d)
        s /= scale_f
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    scores = torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def matching_our(img1_path, img2_path, net, num_features, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr,
        rep_thr=args.repeatability_thr)

    img1 = Image.open(img1_path).convert('RGB')
    W, H = img1.size
    img = TF.to_tensor(img1).unsqueeze(0)
    img = (img - img.mean(dim=[-1, -2], keepdim=True)) / img.std(dim=[-1, -2], keepdim=True)
    img = img.cuda()
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '1',
                                           scale_f=args.scale_f,
                                           min_scale=args.min_scale,
                                           max_scale=args.max_scale,
                                           min_size=args.min_size,
                                           max_size=args.max_size,
                                           verbose=True)
    if len(scores) < num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(num_features)[1]
    kp1 = xys[idxs].cpu().numpy()
    desc1 = desc[idxs].cpu()
    if args.shift_tf:
        if args.single_shift:
            desc1 = shift_topk_candidate(desc1.unsqueeze(0)).squeeze(0)
        else:
            kp1, desc1 = shift_ratio_candidate(xys[idxs].unsqueeze(0), desc1.unsqueeze(0), ratio=args.shift_ratio)
            kp1 = kp1.cpu().numpy()
    img2 = Image.open(img2_path).convert('RGB')
    W, H = img2.size
    img = TF.to_tensor(img2).unsqueeze(0)
    img = (img - img.mean(dim=[-1, -2], keepdim=True)) / img.std(dim=[-1, -2], keepdim=True)
    img = img.cuda()

    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '2',
                                           scale_f=args.scale_f,
                                           min_scale=args.min_scale,
                                           max_scale=args.max_scale,
                                           min_size=args.min_size,
                                           max_size=args.max_size,
                                           verbose=True)
    if len(scores) < num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(num_features)[1]
    kp2 = xys[idxs].cpu().numpy()
    desc2 = desc[idxs].cpu()
    if args.shift_tf:
        if args.single_shift:
            desc2 = shift_topk_candidate(desc2.unsqueeze(0)).squeeze(0)
        else:
            kp2, desc2 = shift_ratio_candidate(xys[idxs].unsqueeze(0), desc2.unsqueeze(0), ratio=args.shift_ratio)
            kp2 = kp2.cpu().numpy()
    # match
    matches, _ = mnn_matcher(desc1, desc2)
    src_pts = kp1[matches[:, 0], :2]  # .cpu().numpy()
    dst_pts = kp2[matches[:, 1], :2]  # .cpu().numpy()
    return src_pts, dst_pts


def get_p_m(ncm_array, rmse_array):
    ncm_array = ncm_array.flatten()
    rmse_array = rmse_array.flatten()
    mask_greater_than_10 = ncm_array > 10
    proportion = sum(mask_greater_than_10) / len(rmse_array)
    # 计算大于10的元素的平均值
    mean_ncm = np.mean(ncm_array[mask_greater_than_10])
    # 获取大于10的元素的索引
    indices_greater_than_10 = np.where(mask_greater_than_10)[0]
    mean_rmse = np.mean(rmse_array[indices_greater_than_10])
    return proportion, mean_ncm, mean_rmse

def parse_args():
    parser = argparse.ArgumentParser("REMM inference demo")
    parser.add_argument("--model-name", type=str, default="REMM")
    parser.add_argument("--model-path", type=str, default="Pretrained/SAR2/50.pt")
    parser.add_argument("--img1", type=str, default="data/SAR2/opt_10_0_11.png")
    parser.add_argument("--img2", type=str, default="data/SAR2/sar_10_0_11.png")
    parser.add_argument("--homography-file", type=str, default="data/SAR2/gt_10_0_11.txt")
    parser.add_argument("--gpu", type=int, default=0, help="use -1 for CPU")
    parser.add_argument("--num-features", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thres", type=int, default=3)
    parser.add_argument("--ransac", action="store_true")
    parser.add_argument("--shift-tf", dest="shift_tf", action="store_true")
    parser.add_argument("--no-shift-tf", dest="shift_tf", action="store_false")
    parser.set_defaults(shift_tf=True)
    parser.add_argument("--single-shift", action="store_true", default=False)
    parser.add_argument("--shift-ratio", type=float, default=0.1)
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=128)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--reliability-thr", type=float, default=0.5)
    parser.add_argument("--repeatability-thr", type=float, default=0.4)
    parser.add_argument("--output-prefix", type=str, default="True")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)

    h = None
    if args.homography_file:
        try:
            h = np.loadtxt(args.homography_file)
        except Exception as exc:
            raise RuntimeError(f"failed to load homography file: {args.homography_file}") from exc

    net = load_network(args.model_name, args.model_path)
    net = net.cuda()
    image_dir1 = args.img1
    image_dir2 = args.img2
    src_pts, dst_pts = matching_our(image_dir1, image_dir2, net, num_features=args.num_features, args=args)
    thres = args.thres
    if args.ransac:
        new_vp1, new_vp2 = rancac(dst_pts, src_pts)
    else:
        if h is not None:
            rmse, NCM, new_vp1, new_vp2 = c_rmse_NCM_NewKP_thres(dst_pts, src_pts, h, thres)
        else:
            new_vp1, new_vp2 = rancac(dst_pts, src_pts)

    combined_array = np.hstack((np.array(new_vp1), np.array(new_vp2)))
    combined_array = np.unique(combined_array, axis=0)
    new_vp1 = combined_array[:, :2]
    new_vp2 = combined_array[:, 2:]
    if h is not None:
        rmse = c_rmse(new_vp1, new_vp2, h)
    NCM = len(new_vp1)
    np.savetxt(f"{args.output_prefix}_{thres}.txt", combined_array, fmt='%f', delimiter=' ')
    if h is not None:
        print(f"rmse: {rmse:.4f}, NCM: {NCM}")
    else:
        print(f"NCM: {NCM}")
    img1 = cv2.imread(image_dir1) ##OPT
    img2 = cv2.imread(image_dir2) ##SAR
    savevis(img2, img1, new_vp1, new_vp2, f"{args.output_prefix}_{thres}.png")
if __name__ == '__main__':
    main()


