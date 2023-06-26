import os
import pickle5 as pickle
import pytorch3d.transforms as py_t
import torch
import numpy as np
from evaluation.eval_utils import compute_RT_errors
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open3d as o3d
from scipy.spatial import KDTree

from sklearn.metrics.pairwise import cosine_similarity

def get_cos_sim(PC,PC_idx,target_PC,target_PC_idx):
    PC = PC[PC_idx].detach().cpu().numpy()
    target_PC = target_PC[target_PC_idx].detach().cpu().numpy()
    sim_li = []
    for i in range(PC.shape[0]):
        sim = cosine_similarity(PC[i],target_PC[i])
        sim_li.append(np.max(sim))
        
    return np.mean(sim_li)

def get_neighbor_index(vertices: "(vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    vertices = torch.from_numpy(vertices)
    inner = torch.mm(vertices, vertices.transpose(0, 1))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=1)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(0) + quadratic.unsqueeze(1)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, 1:]
    return neighbor_index
def indexing_neighbor_new(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    bs, num_points, num_dims = tensor.size()
    idx_base = torch.arange(0, bs, device=tensor.device).view(-1, 1, 1) * num_points
    idx = index + idx_base
    idx = idx.view(-1)
    feature = tensor.reshape(bs * num_points, -1)[idx, :]
    _, out_num_points, n = index.size()
    feature = feature.view(bs, out_num_points, n, num_dims)
    return feature

def get_neighbor_indices(origin_pcl, target_pcl, k=20):
    tree = KDTree(origin_pcl)
    dists, indices = tree.query(target_pcl, k=k)
    return indices
def get_closest_points(A, B):
    tree = KDTree(B)
    dists, indices = tree.query(A)
    closest_points = np.take(B, indices, axis=0)
    return indices
def get_target_neighbor_index(vertices: "(vertice_num, 3)", neighbor_index, target_vertices, neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    vertices = torch.from_numpy(vertices)
    target_vertices = torch.from_numpy(target_vertices)
    inner = torch.matmul(vertices, target_vertices.transpose(0, 1))  # (bs, v, v)
    origin_quadratic = torch.sum(vertices ** 2, dim=1)  # (bs, v)
    target_quadratic = torch.sum(target_vertices ** 2, dim=1)  # (bs, v)
    distance = origin_quadratic.unsqueeze(1) + -2 * inner + target_quadratic.unsqueeze(0)
    neighbor_index = torch.topk(distance, k=neighbor_num+1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, 1:]
    return neighbor_index
def get_neighbor_indices(B, closest_indices, k=20):
    tree = KDTree(B)
    neighbor_indices = []
    for indices in closest_indices:
        dists, neighbors = tree.query(B[indices], k=k+1)
        neighbor_indices.append(neighbors[1:])  # Exclude the first index itself
    return np.array(neighbor_indices)
def calculate_pca(data, num_components):
    # Create PCA object with the desired number of components
    pca = PCA(n_components=num_components)

    # Fit the PCA model to the data
    pca.fit(data)

    # Transform the data to the principal components
    transformed_data = pca.transform(data)

    # Return the transformed data and the explained variance ratio
    return transformed_data, pca.explained_variance_ratio_

def calculate_cosine_similarity(point_cloud1, point_cloud2):
    # Step 2: Compute the centroids
    centroid1 = np.mean(point_cloud1, axis=0)
    centroid2 = np.mean(point_cloud2, axis=0)

    # Step 3: Center the point clouds
    centered_cloud1 = point_cloud1 - centroid1
    centered_cloud2 = point_cloud2 - centroid2

    # Step 4: Compute the covariance matrices
    # covariance1 = np.cov(centered_cloud1.T)
    # covariance2 = np.cov(centered_cloud2.T)

    # # Step 5: Compute the cosine similarity
    # similarity = np.sum(covariance1 * covariance2) / (np.sqrt(np.sum(covariance1**2)) * np.sqrt(np.sum(covariance2**2)))
    similarity = cosine_similarity(centered_cloud1, centered_cloud2).mean()
    return similarity

def cal_rot(sRT_1,sRT_2):
    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])
    return result
def origin_file_open(idx):
    with open('output/results/origin/data_{}.pickle'.format(idx),'rb') as f:
        data = pickle.load(f)
    with open('output/results/origin/output_{}.pickle'.format(idx),'rb') as f:
        output_dict = pickle.load(f)
    return data, output_dict
def rgb_file_open(idx):
    with open('output/results/rgb/data_{}.pickle'.format(idx),'rb') as f:
        data = pickle.load(f)
    with open('output/results/rgb/output_{}.pickle'.format(idx),'rb') as f:
        output_dict = pickle.load(f)
    return data, output_dict

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

origin = '/home/choisj/git/sj/HS-Pose/data/Real/test/scene_1/0000_label.pkl'
base = '/home/choisj/git/sj/HS-Pose/data/Real/test/scene_1'
target1 = '/home/choisj/git/sj/HS-Pose/data/Real/test/scene_1/0001_label.pkl'
target2 = '/home/choisj/git/sj/HS-Pose/data/Real/test/scene_1/0138_label.pkl'
with open(origin, 'rb') as f:
    origin_f = pickle.load(f)
with open(target1, 'rb') as f:
    target_f1 = pickle.load(f)
with open(target2, 'rb') as f:
    target_f2 = pickle.load(f)
    
pcd = o3d.geometry.PointCloud()
def rot_trans_cosine(name):
    if name == 'origin':
        data,output_dict = origin_file_open(1)
    else:
        data,output_dict = rgb_file_open(1)
        
    LAPTOP_INDEX = 3
    PC = data['pcl_in'][LAPTOP_INDEX][:,:3].detach().cpu().numpy()
    translations = origin_f['translations'][2]
    rotations = origin_f['rotations'][2]
    PC = np.dot(PC[:,] - translations,rotations.T)
    # pcd.points = o3d.utility.Vector3dVector(PC) # Origin PCL
    # o3d.visualization.draw_geometries([pcd])
    rot_li = []
    trans_li = []
    cosine_sim_li = []
    origin_neighbor_idx = get_neighbor_index(PC,5)
    for i in range(2,380):
        with open('output/results/{}/output_{}.pickle'.format(name,i),'rb') as f1:
            target_output_dict = pickle.load(f1)
        with open('output/results/{}/data_{}.pickle'.format(name,i),'rb') as f2:
            target_data_dict = pickle.load(f2)
        if not os.path.isfile(os.path.join(base,str(i).zfill(4)+'_label.pkl')):
            continue
        with open(os.path.join(base,str(i).zfill(4)+'_label.pkl'),'rb') as f3:
            target_gts = pickle.load(f3)
        target_class_idx = 3
        if target_class_idx not in target_gts['class_ids']:
            continue
        real_idx = np.where(target_gts['class_ids'] == target_class_idx)
        if real_idx[0][0] > target_output_dict['feat'].shape[0] or real_idx[0][0] == target_output_dict['feat'].shape[0]:
            continue
        # target_value = target_output_dict['feat'][real_idx[0][0]].detach().cpu().numpy().reshape((-1,1))
        # target_value = target_value / np.linalg.norm(target_value)
        target_idx = np.where(target_data_dict['cat_id'] == target_class_idx)
        if np.size(target_idx) == 0:
            continue
        if target_idx[0][0] > target_data_dict['pcl_in'].shape[0] or target_idx[0][0] == target_data_dict['pcl_in'].shape[0]:
            continue
        target_PC = target_data_dict['pcl_in'][target_idx[0][0]][:,:3].detach().cpu().numpy()
        target_translations = target_gts['translations'][real_idx[0][0]]
        target_rotations = target_gts['rotations'][real_idx[0][0]]
        target_PC = np.dot(target_PC[:,]-target_translations,target_rotations.T)
        ########target_neighbor_idx = get_target_neighbor_index(PC,origin_neighbor_idx,target_PC,20)
        
        target_neighbor_idx = get_closest_points(PC,target_PC)
        target_neighbor_idx = get_neighbor_indices(target_PC,target_neighbor_idx,k=5)
        # pcd.points = o3d.utility.Vector3dVector(target_PC) # Origin PCL
        # o3d.visualization.draw_geometries([pcd])
        
        # for j in range(10):
        #     pcd.points = o3d.utility.Vector3dVector(PC) # Origin PCL
        #     color = torch.zeros_like(torch.from_numpy(PC))
        #     color[origin_neighbor_idx[j,:]] = torch.Tensor([255,0,0])
        #     pcd.colors = o3d.utility.Vector3dVector(color) # Origin PCL
        #     o3d.visualization.draw_geometries([pcd])
            
        #     pcd.points = o3d.utility.Vector3dVector(target_PC) # Origin PCL
        #     color = torch.zeros_like(torch.from_numpy(target_PC))
        #     color[target_neighbor_idx[j,:]] = torch.Tensor([255,0,0])
        #     pcd.colors = o3d.utility.Vector3dVector(color) # Origin PCL
        #     o3d.visualization.draw_geometries([pcd])
        ###############3
        #     pcd.points = o3d.utility.Vector3dVector(PC) # Origin PCL
        #     o3d.visualization.draw_geometries([pcd])
        cosine_sim = get_cos_sim(output_dict['feat'][2],origin_neighbor_idx,target_output_dict['feat'][real_idx[0][0]],target_neighbor_idx)
        ################
        # cosine_sim = cosine_similarity(output_dict['feat'][4].detach().cpu().numpy())
        # cosine_sim = cosine_similarity(output_dict['feat'][LAPTOP_INDEX].detach().cpu().numpy().reshape(1028,-1),target_output_dict['feat'][real_idx[0][0]].detach().cpu().numpy().reshape(1028,-1))
        # cosine_sim = np.mean(np.mean(cosine_sim,axis=1))
        # print(cosine_sim)
        # print(explaine_variance)
        # target_pc = 
        rot,trans = cal_rot(origin_f['poses'][2],target_gts['poses'][real_idx[0][0]])
        # print('R: {0: >5.3f}, T: {1: >5.3f}, Cos: {2: >5.3f}'.format(rot,trans,cosine_sim))
        rot_li.append(rot)
        trans_li.append(trans)
        cosine_sim_li.append(cosine_sim)
        print(cosine_sim)
        f1.close()
        f2.close()
    return rot_li,trans_li,cosine_sim_li

rgb_rot_li,rgb_trans_li,rgb_cosine_sim_li = rot_trans_cosine('rgb')
# rgb_rot_li,rgb_trans_li,rgb_cosine_sim_li = rot_trans_cosine('rgb')
# rgb_rot_li,rgb_trans_li,rgb_cosine_sim_li = rot_trans_cosine('rgb')
plt.scatter(rgb_rot_li,rgb_cosine_sim_li,c='r')
data_matrix = np.vstack((rgb_rot_li,rgb_cosine_sim_li))
cov_mat = np.cov(data_matrix)
a,b = np.linalg.eig(cov_mat)
max_idx=np.argmax(a)
max_eigenvector=b[:,max_idx]
center_x = np.mean(rgb_rot_li)
center_y = np.mean(rgb_cosine_sim_li)
plt.quiver(center_x,center_y,max_eigenvector[0],max_eigenvector[1],angles='xy',scale_units='xy',scale=0.0001, color='b')
plt.quiver(center_x,center_y,-max_eigenvector[0],-max_eigenvector[1],angles='xy',scale_units='xy',scale=0.0001, color='b')
    
print()
# plt.plot(rgb_rot_li,rgb_cosine_sim_li)