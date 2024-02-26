import argparse
import os
import numpy as np
import torch
import sliceFormer_model, sliceFormer_model_high_resolution
torch.cuda.current_device()
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data.distributed
from tqdm import tqdm
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss
from utils import RunningAverage
import torch.nn.functional as F
import cv2

PROJECT = "SliceNet"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    if args.resolution == 'high':
        model = sliceFormer_model_high_resolution.SliceFormer.build(100)
    else:
        model = sliceFormer_model.SliceFormer.build(100)
    loaded_state_dict = torch.load(args.trained_model)
    model.load_state_dict(loaded_state_dict)

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    trained_model = train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.model, optimizer_state_dict=None)

def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    test_loader = DepthDataLoader(args, 'online_eval').data

    model.eval()
    criterion_ueff = SILogLoss()
    epoch = 0
    metrics, val_si = validate(args, model, test_loader, epoch, epochs, device)
    print("Validated: {}".format(metrics))


def validate(args, model, test_loader, epoch, epochs, device='cpu'):
    l1_criterion = nn.L1Loss()
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = utils.RunningAverageDict()
        for i, batch in tqdm(enumerate(test_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"):
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

            bin_edges, pred = model(img)

            if args.resolution == 'high':
                depth = F.interpolate(depth, size=(512, 1024), mode='bilinear', align_corners=False)
            else:
                depth = F.interpolate(depth, size=(128, 256), mode='bilinear', align_corners=False)
            l_dense = l1_criterion(pred, depth)
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear',
                                             align_corners=True)
            val_si.append(l_dense.item())

            pred = pred.squeeze().cpu().numpy()

            saved_path = args.saved_data_path + test_loader.dataset.filenames[i].split()[1].replace('.exr', '_output.exr')
            if not os.path.exists(os.path.dirname(saved_path)):
                os.makedirs(os.path.dirname(saved_path))
            cv2.imwrite(saved_path, pred)

            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            if utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]) is None:
                print()
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si



if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="SliceNet")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    parser.add_argument("--model", default='slicemViT2', type=str, help="Model to train")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--comments", default='', type=str, help="comments for the name of saved model")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='3D60', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='', type=str, help="path to dataset")
    parser.add_argument("--saved_data_path", default='', type=str, help="path to saved outputs")
    parser.add_argument('--filenames_file_eval', default='', type=str,
                        help='Stanford2D3D 5p testing dataset (82 images)')
    parser.add_argument('--trained_model', default='', type=str,
                        help='path of trained model')
    parser.add_argument('--resolution', default='', type=str,
                        help='output resolution of trained model (high/low)')
    parser.add_argument('--input_height', type=int, help='input height', default=256)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--lstm_direction', type=str, help='minimum depth in estimation', default='combined')
    parser.add_argument('--do_random_rotate', default=False,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')
    parser.add_argument('--data_path_eval',
                        default="/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--logging', type=bool, help='Whether log the log to WandB', default=False)
    parser.add_argument('--run_times', type=int, help='Run times for uncertainty', default=1)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

