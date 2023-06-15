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
    

def rot_trans_cosine(name):
    if name == 'origin':
        _,output_dict = origin_file_open(1)
    else:
        _,output_dict = rgb_file_open(1)
        
    rot_li = []
    trans_li = []
    cosine_sim_li = []
    for i in range(2,380):
        with open('output/results/{}/output_{}.pickle'.format(name,i),'rb') as f1:
            target_output_dict = pickle.load(f1)
        if not os.path.isfile(os.path.join(base,str(i).zfill(4)+'_label.pkl')):
            continue
        with open(os.path.join(base,str(i).zfill(4)+'_label.pkl'),'rb') as f2:
            target_f = pickle.load(f2)
        target_class_idx = 5
        if target_class_idx not in target_f['class_ids']:
            continue
        real_idx = np.where(target_f['class_ids'] == target_class_idx)
        if real_idx[0][0] > target_output_dict['feat'].shape[0] or real_idx[0][0] == target_output_dict['feat'].shape[0]:
            continue
        # target_value = target_output_dict['feat'][real_idx[0][0]].detach().cpu().numpy().reshape((-1,1))
        # target_value = target_value / np.linalg.norm(target_value)
        cosine_sim = cosine_similarity(output_dict['feat'][4].detach().cpu().numpy().reshape(1028,-1),target_output_dict['feat'][real_idx[0][0]].detach().cpu().numpy().reshape(1028,-1))
        cosine_sim = np.mean(np.mean(cosine_sim,axis=1))
        # print(cosine_sim)
        
        # print(explaine_variance)
        rot,trans = cal_rot(origin_f['poses'][4],target_f['poses'][real_idx[0][0]])
        # print('R: {0: >5.3f}, T: {1: >5.3f}, Cos: {2: >5.3f}'.format(rot,trans,cosine_sim))
        rot_li.append(rot)
        trans_li.append(trans)
        cosine_sim_li.append(cosine_sim)
        f1.close()
        f2.close()
    return rot_li,trans_li,cosine_sim_li

rot_li,trans_li,cosine_sim_li = rot_trans_cosine('origin')
rgb_rot_li,rgb_trans_li,rgb_cosine_sim_li = rot_trans_cosine('rgb')
plt.scatter(rot_li,cosine_sim_li)
data_matrix = np.vstack((rot_li,cosine_sim_li))
cov_mat = np.cov(data_matrix)
a,b = np.linalg.eig(cov_mat)
max_idx=np.argmax(a)
max_eigenvector=b[:,max_idx]
center_x = np.mean(rot_li)
center_y = np.mean(cosine_sim_li)
plt.quiver(center_x,center_y,max_eigenvector[0],max_eigenvector[1],angles='xy',scale_units='xy',scale=0.0001, color='r')
plt.quiver(center_x,center_y,-max_eigenvector[0],-max_eigenvector[1],angles='xy',scale_units='xy',scale=0.0001, color='r')
    
print()
plt.plot(rot_li,cosine_sim_li)