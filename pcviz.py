import pptk
import pickle5 as pickle
# import pickle
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
# import open3d as o3d

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def calc_cos_sim(origin,target1,target2):
    min = 100000
    # origin = [-0.322, -0.115, 0.972]
    # target1 = [-0.182, -0.162, 1.048]
    # target2 = [-0.070, 0.024, 0.957]
    origin_index = -1
    target_index1 = -1
    target_index2 = -1
    for i, point in enumerate(x):
        cal = np.sum((point - origin)**2)
        if min > cal:
            origin_index = i
            min = cal

    min = 100000
    for i, point in enumerate(x):
        cal = np.sum((point - target1)**2)
        if min > cal:
            target_index1 = i
            min = cal
            
    min = 100000
    for i, point in enumerate(x):
        cal = np.sum((point - target2)**2)
        if min > cal:
            target_index2 = i
            min = cal
          
    origin_feat = output_dict['feat'][SEE][origin_index].detach().cpu().numpy()
    target1_feat = output_dict['feat'][SEE][target_index1].detach().cpu().numpy()
    target2_feat = output_dict['feat'][SEE][target_index2].detach().cpu().numpy()

    first = cos_sim(origin_feat,target1_feat)
    second = cos_sim(origin_feat,target2_feat)

    return first, second
  
def calc_all_cos_sim(origin):
    # origin = [-0.322, -0.115, 0.972]
    # target1 = [-0.182, -0.162, 1.048]
    # target2 = [-0.070, 0.024, 0.957]
    ret = []
    origin_index = -1
    min = 100000
    for i, point in enumerate(x):
        cal = np.sum((point - origin)**2)
        if min > cal:
            origin_index = i
            min = cal
    origin_feat = output_dict['feat'][SEE][origin_index].detach().cpu().numpy()
    for i in range(output_dict['feat'][SEE].shape[0]):
        target_feat = output_dict['feat'][SEE][i].detach().cpu().numpy()
        ret.append(cos_sim(origin_feat,target_feat))
        
    return ret

def rgb_file_open(idx):
    with open('output/results/rgb/data_{}.pickle'.format(idx),'rb') as f:
        data = pickle.load(f)
    with open('output/results/rgb/output_{}.pickle'.format(idx),'rb') as f:
        output_dict = pickle.load(f)
    return data, output_dict

def origin_file_open(idx):
    with open('output/results/origin/data_{}.pickle'.format(idx),'rb') as f:
        data = pickle.load(f)
    with open('output/results/origin/output_{}.pickle'.format(idx),'rb') as f:
        output_dict = pickle.load(f)
    return data, output_dict
    
if __name__ == '__main__':
    data, output_dict = rgb_file_open(174)
    # data, output_dict = origin_file_open(179)

    SEE = 3
    x = data['pcl_in'][SEE][:,:3].detach().cpu().numpy()
    x_color = data['pcl_in'][SEE][:,3:].detach().cpu().numpy()*200.0
    # origin = [-0.322, -0.115, 0.972] #1
    # origin = [-0.180, -0.144, 1.048] #3
    # origin = [-0.203, 0.092, 0.841] #2
    # origin = [-0.226, -0.097, 1.017] #4
    origin = [-0.113,-0.127,1.000]

    ret = calc_all_cos_sim(origin)

    v = pptk.viewer(x,ret)
    # v = pptk.viewer(x)#,ret)
    v.set(point_size=0.001)
    
    selected_indexes = v.get('selected')
    print(selected_indexes)
    
    ret = np.array(ret)
    selected_indexes = np.array(selected_indexes)
    ret = ret[selected_indexes]
    print(np.average(ret))