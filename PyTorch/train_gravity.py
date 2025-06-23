import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import numpy as np
import torch
import biLSTM_direct_model, biLSTM_split_model
torch.cuda.current_device()
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm
import utils
from dataloader import DepthDataLoader
from loss_new import SILogLoss
from utils import RunningAverage, colorize
import matplotlib
import time
from loss import BinsChamferLoss

PROJECT = "Gravity"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def is_rank_zero(args):
    return args.rank == 0

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    if args.model == 'biLSTM_direct':
        lstm_hidden_size = 128
        model = biLSTM_direct_model.BiLSTMImage(lstm_hidden_size, args.lstm_direction)
    elif args.model == 'biLSTM_split' or args.model == 'biLSTM_square' or args.model == 'biLSTM_split_rotate':
        lstm_hidden_size = 128
        model = biLSTM_split_model.BiLSTMImage(lstm_hidden_size, args.lstm_direction)

    ################################################################################################

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

    # save the trained-model
    save_path = args.tm + \
                'Ubuntu_' + \
                time.strftime("%Y_%m_%d-%H_%M_UK") \
                + '_epochs-' + str(args.epochs) \
                + '_lr-' + str(args.lr) \
                + '_bs-' + str(args.bs) \
                + '_maxDepth-' + str(args.comments) \
                + '.pth'
    torch.save(trained_model.state_dict(), save_path)
    print('saved the model to:', save_path)

def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and args.logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        # wandb.watch(model)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    params = model.parameters()
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    l2_criterion = nn.MSELoss()
    print('MSELoss!')

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    criterion_bins = BinsChamferLoss()
    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            # remove outliers
            depth[depth < 0.01] = 0.01
            depth[depth > 10] = 10

            # transpose the tensor
            depth = depth.permute(0, 3, 1, 2) # torch.Size([8, 1, 256, 512])

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            if args.model == 'biLSTM_split':
                pred = model(img[: , :, :, :256])
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth[:, :, :, :256])
            elif args.model == 'biLSTM_split_rotate':
                pred = model(img[:, :, :, :256].transpose(2, 3).flip(2))
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth[:, :, :, :256].transpose(2, 3).flip(2))
            else:
                pred = model(img)
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth)

            loss = l_dense
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
            step += 1
            scheduler.step()

            ########################################################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                model.train()
                #################################################################################################

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    l2_criterion = nn.L1Loss()
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

            if args.model == 'biLSTM_split':
                pred = model(img[:, :, :, :256])
            elif args.model == 'biLSTM_split_rotate':
                pred = model(img[:, :, :, :256].transpose(2, 3).flip(2))
            else:
                pred = model(img)

            mask = depth > args.min_depth

            if args.model == 'biLSTM_split':
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth[:, :, :, :256])
                pred = nn.functional.interpolate(pred.unsqueeze(dim=1), depth[:, :, :, :256].shape[-2:], mode='bilinear',
                                                 align_corners=True)
            elif args.model == 'biLSTM_split_rotate':
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth[:, :, :, :256].transpose(2, 3).flip(2))
                pred = nn.functional.interpolate(pred.unsqueeze(dim=1), depth[:, :, :, :256].transpose(2, 3).flip(2).shape[-2:], mode='bilinear',
                                                 align_corners=True)
            else:
                l_dense = l2_criterion(pred.unsqueeze(dim=1), depth)
                pred = nn.functional.interpolate(pred.unsqueeze(dim=1), depth.shape[-2:], mode='bilinear',
                                                 align_corners=True)
            # l_dense = criterion_ueff(pred.unsqueeze(dim=1), depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            if args.model == 'biLSTM_split':
                gt_depth = depth[:,:,:,:256].squeeze().cpu().numpy()
            elif args.model == 'biLSTM_split_rotate':
                gt_depth = depth[:,:,:,:256].transpose(2, 3).flip(2).squeeze().cpu().numpy()
            else:
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


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
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
    parser.add_argument("--model", default='biLSTM_split_rotate', type=str, help="Model to train")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--comments", default='', type=str, help="comments for the name of saved model")
    parser.add_argument("--tm", default='', type=str, help="saved path for trained model")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='3D60', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='', type=str,
                        help="path to dataset")
    parser.add_argument('--filenames_file', default='', type=str,
                        help='Stanford2D3D 5p ALL training dataset')
    parser.add_argument('--filenames_file_eval', default='', type=str,
                        help='Stanford2D3D 5p testing dataset (82 images)')
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


    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    print('The model is ', args.model)

    for run_time in range(args.run_times):

        # show the training and testing datasets
        print('The training dataset is ', args.filenames_file.split('/')[-1])
        print('The testing dataset is ', args.filenames_file_eval.split('/')[-1])

        args.batch_size = args.bs
        args.num_threads = args.workers
        args.mode = 'train'
        args.chamfer = args.w_chamfer > 0
        if args.root != "." and not os.path.isdir(args.root):
            os.makedirs(args.root)

        try:
            node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
            nodes = node_str.split(',')

            args.world_size = len(nodes)
            args.rank = int(os.environ['SLURM_PROCID'])

        except KeyError as e:
            # We are NOT using SLURM
            args.world_size = 1
            args.rank = 0
            nodes = ["127.0.0.1"]

        if args.distributed:
            mp.set_start_method('forkserver')

            print(args.rank)
            port = np.random.randint(15000, 15025)
            args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
            print(args.dist_url)
            args.dist_backend = 'nccl'
            args.gpu = None

        ngpus_per_node = torch.cuda.device_count()
        args.num_workers = args.workers
        args.ngpus_per_node = ngpus_per_node

        if args.distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            if ngpus_per_node == 1:
                args.gpu = 0
            main_worker(args.gpu, ngpus_per_node, args)
