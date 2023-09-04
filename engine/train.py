import os
import random

import math
import torch
from absl import app

from config.config import *
from tools.training_utils import build_lr_rate, build_optimizer
from network.HSPose import HSPose 
import argparse
import yaml

from datasets.load_data import PoseDataset
from tqdm import tqdm
import time
import numpy as np

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger
from tensorflow.compat.v1 import Summary

# for distributed training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

def train():
    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_num']
    ngpus_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    cfg['distributed'] = cfg['world_size'] > 1 or cfg['multiprocessing_distributed']
    if cfg['multiprocessing_distributed']:
        cfg['world_size']=ngpus_per_node*cfg['world_size']
        mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,cfg))
    else:
        main_worker(0, ngpus_per_node, cfg)

def main_worker(gpu,ngpus_per_node,cfg):
    cfg['gpu'] = gpu
    device = cfg["device"]
    
    if cfg['gpu'] is not None:
        print("Use GPU: {} for training".format(cfg['gpu']))
        
    if cfg['distributed']:
        if cfg['dist_url']=='env://' and cfg['rank']==-1:
            cfg['rank']=int(os.environ["RANK"])
        if cfg['multiprocessing_distributed']:
            # gpu = 0,1,2,...,ngpus_per_node-1
            cfg['rank']=cfg['rank']*ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=cfg['dist_backend'],init_method=cfg['dist_url'],
                                            world_size=cfg['world_size'],rank=cfg['rank'])
        
        if torch.distributed.get_rank() == 0:
            print("RANK: {}, WORLD_SIZE: {}".format(cfg['rank'],cfg['world_size']))
    if cfg['resume']:
        checkpoint = torch.load(cfg['resume_model'])
        if 'seed' in checkpoint:
            seed = checkpoint['seed']
        else:
            seed = int(time.time()) if cfg['seed'] == -1 else cfg['seed']
    else:
        seed = int(time.time()) if cfg['seed'] == -1 else cfg['seed']
    seed_init_fn(seed) 
    if not os.path.exists(cfg['model_save']):
        os.makedirs(cfg['model_save'])
    tf.compat.v1.disable_eager_execution()
    tb_writter = tf.compat.v1.summary.FileWriter(cfg['model_save'])
    logger = setup_logger('train_log', os.path.join(cfg['model_save'], 'log.txt'))
    for key, value in cfg.items():
        logger.info(key + ':' + str(value))
    Train_stage = 'PoseNet_only'
    network = HSPose(cfg,Train_stage)
    param_list = network.build_params(training_stage_freeze=[])
    train_steps = cfg["train_steps"]
    
    if cfg["distributed"]:
        if cfg["gpu"] is not None:
            torch.cuda.set_device(cfg["gpu"])
            torch.distributed.barrier()
            network.cuda(cfg["gpu"])
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network) # batch norm mode convert
            cfg["batch_size"] = int(cfg["batch_size"]/ngpus_per_node)
            cfg["num_workers"] = int((cfg["num_workers"]+ngpus_per_node-1)/ngpus_per_node)
            network = DDP(network,device_ids=[cfg["gpu"]],find_unused_parameters=True)
            scheduler = build_lr_rate(optimizer, train_steps * cfg["total_epoch"] // cfg["accumulate"] // ngpus_per_node, cfg)
            
        else:
            network.cuda()
            network = DDP(network,find_unused_parameters=True)
            
    elif cfg["gpu"] is not None:
        torch.cuda.set_device(cfg["gpu"])
        network = network.cuda(cfg["gpu"])
        scheduler = build_lr_rate(optimizer, train_steps * cfg["total_epoch"] // cfg["accumulate"], cfg)
        
        
    #  build optimizer
    optimizer = build_optimizer(param_list,cfg)
    optimizer.zero_grad()   # first clear the grad
    # resume or not
    s_epoch = 0
    if cfg["resume"]:
        # checkpoint = torch.load(FLAGS.resume_model)
        network.load_state_dict(checkpoint['posenet_state_dict'])
        s_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Checkpoint loaded:", checkpoint.keys())


    # build dataset annd dataloader
    train_dataset = PoseDataset(cfg,source=cfg["dataset"], mode='train',
                                data_dir=cfg["dataset_dir"], per_obj=cfg["per_obj"])
    
    if cfg["distributed"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                                   num_workers=cfg["num_workers"], pin_memory=True,
                                                   prefetch_factor = 4,
                                                   worker_init_fn =seed_worker,
                                                   shuffle=(train_sampler is None),
                                                   sampler=train_sampler)
    network.train()
    global_step = train_steps * s_epoch  # record the number iteration
    for epoch in range(s_epoch, cfg["total_epoch"]):
        if cfg["distributed"]:
            train_sampler.set_epoch(epoch)
        i = 0
        for data in tqdm(train_dataloader, desc=f'Training {epoch}/{cfg["total_epoch"]}', dynamic_ncols=True):
            # torch.cuda.synchronize()
            output_dict, loss_dict \
                = network(
                          obj_id=data['cat_id'].to(device), 
                          PC=data['pcl_in'].to(device),
                          rgb=data['rgb_in'].to(device),
                          depth=data['depth_in'].to(device),
                          depth_mask=data['depth_mask'].to(device),
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
                                + depth_loss

            if math.isnan(total_loss):
                print('Found nan in total loss')
                i += 1
                global_step += 1
                continue
            # backward
            if global_step % cfg["accumulate"] == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            global_step += 1
            if i % cfg["log_every"] == 0:
                write_to_summary(tb_writter, optimizer, total_loss, depth_loss, fsnet_loss, prop_loss, recon_loss, global_step)
            i += 1

        # save model
        if (epoch + 1) % cfg["save_every"] == 0 or (epoch + 1) == cfg["total_epoch"]:
            torch.save(
                {
                'seed': seed,
                'epoch': epoch,
                'posenet_state_dict': network.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                '{0}/model_{1:02d}.pth'.format(cfg["model_save"], epoch))
        # torch.cuda.empty_cache()

def write_to_summary(writter, optimizer, total_loss, depth_loss, fsnet_loss, prop_loss, recon_loss, global_step):
    summary = Summary(
        value=[
            Summary.Value(tag='lr', simple_value=optimizer.param_groups[0]["lr"]),
            Summary.Value(tag='train_loss', simple_value=total_loss),
            Summary.Value(tag='depth_loss', simple_value=depth_loss),
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
    train()
