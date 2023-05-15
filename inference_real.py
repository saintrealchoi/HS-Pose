import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
import _pickle as cPickle
import sys
import time

from network.HSPose import HSPose 

import torchvision.transforms as transforms
from utils import camera
from lib.utils import draw_detections,align_rotation, transform_coordinates_3d,calculate_2d_projections,get_3d_bbox
from utils.viz_utils import save_projected_points,line_set_mesh,draw_bboxes,draw_axes,draw_bboxes_origin,draw_2d_bboxes
from utils.transform_utils import get_gt_pointclouds,project
from lib.utils import compute_RT_overlaps,compute_sRT_errors,compute_RT_errors
from tools.eval_utils import load_depth, get_bbox
from matplotlib import cm

category = {1:'bottle',2:'bowl',3:'camera',4:'can',5:'laptop',6:'mug'}
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/NOCS', help='data directory')
parser.add_argument('--data', type=str, default='REAL275', help='CAMERA25, REAL275')
parser.add_argument('--n_pts', type=int, default=1028, help='number of foreground points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1028, help='number of vertices in shape priors')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=25, help='max number of epochs to train')
parser.add_argument('--model', type=str, default='results/real/model_50.pth', help='evaluate model')
# parser.add_argument('--result_dir', type=str, default='results/eval_real_2048_128_ae_model_02', help='directory to save train results')
# parser.add_argument('--ae_model', type=str, default='results/ft_real_2048_128_ae/ae_model_02.pth', help='wandb online mode')

mean_shapes = np.load('assets/mean_points_emb.npy')
opt = parser.parse_args()
synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

assert opt.data in ['CAMERA25', 'REAL275']

if opt.data == 'CAMERA25':
    result_dir = 'results/eval_camera'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    temp_dir_path = '_'.join(opt.model.split('/')[-2:])[:-4]
    result_dir = 'results/eval_' + temp_dir_path
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def inference(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list.txt')).read().splitlines()
    
    _CAMERA = camera.NOCS_Real()
    load_path = os.path.join('/home/choisj/git/sj/HS-Pose/output/pseudo/eval_result_model/pred_result.pkl')
    with open(load_path, 'rb') as f:
        results = cPickle.load(f)
    for i, img_path in enumerate(data_path):
        
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png' 
        if not os.path.exists(color_path):
            continue
        img_vis = cv2.imread(color_path)
        
        img_path_parsing = img_full_path.split('/')
        mrcnn_path = os.path.join('data/segmentation_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            'test', img_path_parsing[-2], img_path_parsing[-1]))
            # opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
            
        result = results[i]

        depth_full_path = os.path.join(data_dir, 'Real', img_path)
        depth_path = depth_full_path + '_depth.png'
        depth = load_depth(depth_path)
        depth_norm = cv2.normalize(depth,None,0,1,cv2.NORM_MINMAX, cv2.CV_32F)
        depth_map_gray = (depth_norm * 255).astype(np.uint8)
        depth_map_magma = cm.magma(depth_map_gray)
        depth_map_magma_uint8 = (depth_map_magma * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / f'{i}_depth.png'), depth_map_magma_uint8)
        
        img_vis_2d = draw_2d_bboxes(np.copy(img_vis),mrcnn_result)
        cv2.imwrite(
            str(output_path / f'{i}_image.png'),
            np.copy(np.copy(img_vis_2d))
        )
        
        write_pcd = False
        rotated_pcds = []
        points_2d = []
        box_obb = []
        axes = []
        gt_axes_li = []
        
        num_insts = len(mrcnn_result['gt_class_ids'])
            
        R_errors = []
        T_errors = []
        
        for j in range(len(result['pred_RTs'])):
        # for j in range(num_insts):
            
        #     shape_out = result['pred_shape'][j]
            
            # rotated_pc, rotated_box, pred_size = get_gt_pointclouds(result['pred_RTs'][j],shape_out,camera_model=_CAMERA)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(rotated_pc)
        #     filename_rotated = str(output_path) + '/pcd_rotated'+str(i)+str(j)+'.ply'
        #     if write_pcd:
        #         o3d.io.write_point_cloud(filename_rotated, pcd)
        #     else:
        #         rotated_pcds.append(pcd)

        #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #     T = result['pred_RTs'][j]
        #     mesh_frame = mesh_frame.transform(T)
        #     rotated_pcds.append(mesh_frame)
        #     cylinder_segments = line_set_mesh(rotated_box)
        #     for k in range(len(cylinder_segments)):
        #         rotated_pcds.append(cylinder_segments[k])

                
        #     points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
        #     points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
        #     points_2d_mesh = points_2d_mesh.T
        #     points_2d.append(points_2d_mesh)
        #     #2D output
            # points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
            # points_2d_obb = project(_CAMERA.K_matrix, points_obb)
            # points_2d_obb = points_2d_obb.T
            # box_obb.append(points_2d_obb)
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            sRT = result['pred_RTs'][j]
            transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
            projected_axes = calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3])

            
            axes.append(projected_axes)
        #     #RT output
            
            max_T = 10000
            candidate_idx = -1
            for idx,candidate in enumerate(result['gt_RTs']):
                temp = compute_RT_errors(candidate, sRT, result['pred_class_ids'][j], 1, synset_names)
                if temp[1] < max_T :
                    max_T = temp[1]
                    candidate_idx = idx
                    

            results_rt = compute_RT_errors(result['gt_RTs'][candidate_idx], sRT, result['pred_class_ids'][j], 1, synset_names)
            
            R_errors.append(float(results_rt[0]))
            T_errors.append(float(results_rt[1]))
            
        for k in range(result['gt_RTs'].shape[0]):
            gt_axes = transform_coordinates_3d(xyz_axis,result['gt_RTs'][k])
            gt_projected_axes = calculate_2d_projections(gt_axes,_CAMERA.K_matrix[:3,:3])
            gt_axes_li.append(gt_projected_axes)
            
        if not write_pcd:
        # o3d.visualization.draw_geometries(rotated_pcds)
            # save_projected_points(np.copy(img_vis), points_2d, str(output_path), i, result['pred_class_ids'])
        
            colors_box = [(0,0,220)]
            im = np.array(np.copy(img_vis)).copy()
            
            font=cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            
            count = 0
            for cat in range(1,7):
                if cat not in result['pred_class_ids']:
                    continue
                for x in range(len(result['pred_RTs'])):
                    if cat == result['pred_class_ids'][x]:
                        text = '{0} : R_{1:.3f}, T_{2:.3f}'.format(category[result['pred_class_ids'][x]],R_errors[x],T_errors[x])
                        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                        text_x = im.shape[1] - text_size[0] - 10 # adjust 10 to your preference
                        text_y = im.shape[0] - 10 # adjust 10 to your preference
                        cv2.putText(im,text,(text_x,text_y-15*count),font,font_scale,(255,255,255),font_thickness)
                        count+=1
                        
            # Draw Pred BBoxes, Axes
            # for k in range(len(colors_box)):
            #     for points_2d, axis in zip(box_obb, axes):
            #         points_2d = np.array(points_2d)
            #         im = draw_bboxes(im, points_2d, axis, colors_box[k])
                    
                    
            # Draw Pred BBoxes, Axes
            for k in range(result['pred_RTs'].shape[0]):
                if result['pred_class_ids'][k] in [1, 2, 4]:
                    sRT = align_rotation(result['pred_RTs'][k, :, :])
                else:
                    sRT = result['pred_RTs'][k, :, :]
                bbox_3d = get_3d_bbox(result['pred_scales'][k, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, _CAMERA.K_matrix[:3,:3] )
                im = draw_bboxes_origin(im, projected_bbox,(0, 0, 255))
                
            # Draw GT Axes
            # for k in range(len(colors_box)):
            #     for points_2d, axis in zip(result['gt_bboxes'], gt_axes_li):
            #         points_2d = np.array(points_2d)
            #         im = draw_axes(im, points_2d, axis, colors_box[k])
                    
            # Draw GT BBoxes 
            for k in range(result['gt_RTs'].shape[0]):
                if result['gt_class_ids'][k] in [1, 2, 4]:
                    sRT = align_rotation(result['gt_RTs'][k, :, :])
                else:
                    sRT = result['gt_RTs'][k, :, :]
                bbox_3d = get_3d_bbox(result['gt_scales'][k, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, _CAMERA.K_matrix[:3,:3])
                im = draw_bboxes_origin(im, projected_bbox, (0, 255, 0))
            box_plot_name = str(output_path)+'/box3d'+str(i).zfill(3)+'.png'
            cv2.imwrite(
                box_plot_name,
                np.copy(im)
            )
            cv2.imwrite(str(output_path / f'concat_{i}.png'),np.concatenate((im,depth_map_magma_uint8[:,:,:3],img_vis_2d),axis=1))
            
        print("done with image: ", i )

if __name__ == '__main__':
  print(opt)
  result_name = 'inference'
  path = 'data/'+result_name
  output_path = pathlib.Path(path) / opt.model[-12:-4]
  output_path.mkdir(parents=True, exist_ok=True)
  inference(opt, opt.data_dir, output_path)
