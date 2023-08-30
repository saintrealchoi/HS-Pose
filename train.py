import os
import random

import math
import torch
from absl import app

from config.config import *
from tools.training_utils import build_lr_rate, build_optimizer
from network.HSPose import HSPose 
FLAGS = flags.FLAGS
from datasets.load_data import PoseDataset
from tqdm import tqdm
import time
import numpy as np
import utils.ddp
# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger
from tensorflow.compat.v1 import Summary

import torch.distributed as dist
import torch.multiprocessing as mp
from utils.seed import setup_seed
torch.autograd.set_detect_anomaly(True)
import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="", help="Load weights to resume training")

    parser.add_argument("--dataset", type=str, default="./configs/datasets/voc0712.yaml")
    parser.add_argument("--model", type=str, default="./configs/models/r18_s4.yaml")
    parser.add_argument("--optimizer", type=str, default="./configs/optimizers/base_optimizer.yaml")
    
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers used in dataloading")
    parser.add_argument("--save-folder", default="./weights", type=str, help="Where you save weights")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--seed", default=7777, type=int)

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

def train(args):
    utils.ddp.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)
    
    seed = args.seed + utils.ddp.get_rank()
    setup_seed(seed)
        
    
    
    # if FLAGS.resume:
    #     checkpoint = torch.load(FLAGS.resume_model)
    #     if 'seed' in checkpoint:
    #         seed = checkpoint['seed']
    #     else:
    #         seed = int(time.time()) if FLAGS.seed == -1 else FLAGS.seed
    # else:
    #     seed = int(time.time()) if FLAGS.seed == -1 else FLAGS.seed
    # seed_init_fn(seed) 
    
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    tb_writter = tf.compat.v1.summary.FileWriter(FLAGS.model_save)
    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))
    for key, value in vars(FLAGS).items():
        logger.info(key + ':' + str(value))
    Train_stage = 'PoseNet_only'
    network = HSPose(Train_stage)
    network = network.to(args.device)
    train_dataset = PoseDataset(source=FLAGS.dataset, mode='train',
                            data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    if args.distributed:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.gpu])
        model_without_ddp = network.module
        
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=utils.ddp.get_world_size(), rank=utils.ddp.get_rank(), shuffle=True
            )
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                num_workers=FLAGS.num_workers,
                                                sampler=sampler_train,
                                                pin_memory=True,
                                                prefetch_factor = 4,
                                                worker_init_fn =seed_worker,
                                                shuffle=True )    
        
    train_steps = FLAGS.train_steps    
    #  build optimizer
    param_list = network.build_params(training_stage_freeze=[])
    optimizer = build_optimizer(param_list)
    optimizer.zero_grad()   # first clear the grad
    scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
    # resume or not
    s_epoch = 0
    # if FLAGS.resume:
    #     # checkpoint = torch.load(FLAGS.resume_model)
    #     network.load_state_dict(checkpoint['posenet_state_dict'])
    #     s_epoch = checkpoint['epoch'] + 1
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     print("Checkpoint loaded:", checkpoint.keys())


    # build dataset annd dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                   num_workers=FLAGS.num_workers, pin_memory=True,
                                                   prefetch_factor = 4,
                                                   worker_init_fn =seed_worker,
                                                   shuffle=True )
    network.train()
    global_step = train_steps * s_epoch  # record the number iteration
    for epoch in range(s_epoch, FLAGS.total_epoch):
        i = 0
        for data in tqdm(train_dataloader, desc=f'Training {epoch}/{FLAGS.total_epoch}', dynamic_ncols=True):
            output_dict, loss_dict \
                = network(
                          obj_id=data['cat_id'].to(device),
                          PC=data['pcl_in'].to(device),
                          rgb=data['rgb_in'].to(device),
                          depth=data['depth_in'].to(device),
                          depth_valid = data['depth_valid'].to(device),
                          sample_idx = data['sample_idx'].to(device),
                          gt_R=data['rotation'].to(device), 
                          gt_t=data['translation'].to(device),
                          gt_s=data['fsnet_scale'].to(device), 
                          mean_shape=data['mean_shape'].to(device),
                          sym=data['sym_info'].to(device),
                          aug_bb=data['aug_bb'].to(device), 
                          aug_rt_t=data['aug_rt_t'].to(device),
                          aug_rt_r=data['aug_rt_R'].to(device),
                          model_point=data['model_point'].to(device), 
                          nocs_scale=data['nocs_scale'].to(device),
                          do_loss=True)
            # print('net_process', time.time()-begin)
            fsnet_loss = loss_dict['fsnet_loss']
            recon_loss = loss_dict['recon_loss']
            geo_loss = loss_dict['geo_loss']
            prop_loss = loss_dict['prop_loss']
            depth_loss = loss_dict['depth_loss']

            total_loss = sum(fsnet_loss.values()) + sum(recon_loss.values()) \
                            + sum(geo_loss.values()) + sum(prop_loss.values()) \

            if math.isnan(total_loss):
                print('Found nan in total loss')
                i += 1
                global_step += 1
                continue
            # backward
            if global_step % FLAGS.accumulate == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            global_step += 1
            if i % FLAGS.log_every == 0:
                write_to_summary(tb_writter, optimizer, total_loss, fsnet_loss, prop_loss, recon_loss, global_step)
            i += 1

        # save model
        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(
                {
                'seed': seed,
                'epoch': epoch,
                'posenet_state_dict': network.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))
        torch.cuda.empty_cache()

def write_to_summary(writter, optimizer, total_loss, fsnet_loss, prop_loss, recon_loss, global_step):
    summary = Summary(
        value=[
            Summary.Value(tag='lr', simple_value=optimizer.param_groups[0]["lr"]),
            Summary.Value(tag='train_loss', simple_value=total_loss),
            Summary.Value(tag='rot_loss_1', simple_value=fsnet_loss['Rot1']),
            Summary.Value(tag='rot_loss_2', simple_value=fsnet_loss['Rot2']),
            Summary.Value(tag='T_loss', simple_value=fsnet_loss['Tran']),
            Summary.Value(tag='Prop_sym_recon', simple_value=prop_loss['Prop_sym_recon']),
            Summary.Value(tag='Prop_sym_rt', simple_value=prop_loss['Prop_sym_rt']),
            Summary.Value(tag='Size_loss', simple_value=fsnet_loss['Size']),
            Summary.Value(tag='Face_loss', simple_value=recon_loss['recon_per_p']),
            Summary.Value(tag='Recon_loss_r', simple_value=recon_loss['recon_point_r']),
            Summary.Value(tag='Recon_loss_t', simple_value=recon_loss['recon_point_t']),
            Summary.Value(tag='Recon_loss_s', simple_value=recon_loss['recon_point_s']),
            Summary.Value(tag='Recon_p_f', simple_value=recon_loss['recon_p_f']),
            Summary.Value(tag='Recon_loss_se', simple_value=recon_loss['recon_point_self']),
            Summary.Value(tag='Face_loss_vote', simple_value=recon_loss['recon_point_vote']), ])
    writter.add_summary(summary, global_step)
    return

def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    train(args)
