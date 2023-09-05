import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
	def __init__(self,alpha=10):
		super(SILogLoss, self).__init__()
		self.name = 'SILog'
		self.alpha=alpha

	def forward(self, output,depth_gt):
		losses=[]
		for i in range(len(output)):
			mask_pred = output[i]>0.01
			mask_gt=depth_gt[i]>0.01
			mask=torch.logical_and(mask_pred,mask_gt)
			input = output[i][mask]
			target = depth_gt[i][mask]
			# mask=depth_gt[i]>0.01
			# input = output[i][mask]
			# target = depth_gt[i][mask]
			g = torch.log(input+0.1) - torch.log(target+0.1)
			# n, c, h, w = g.shape
			# norm = 1/(h*w)
			# Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
			Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
			losses.append(self.alpha * torch.sqrt(Dg))
		total_loss=sum(losses)
		return total_loss

# class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
#     def __init__(self):
#         super().__init__()
#         self.name = "ChamferLoss"

#     def forward(self, bins, target_depth_maps):
#         bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
#         n, p = bin_centers.shape
#         input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
#         # n, c, h, w = target_depth_maps.shape
#         target_points = target_depth_maps.flatten(1)  # n, hwc
#         mask = target_points.ge(1e-3)  # only valid ground truth points
#         target_points = [p[m] for p, m in zip(target_points, mask)]
#         target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
#         target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
#         loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
#         return loss

class MinmaxLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "MinMaxLoss"
	def forward(self, bins, target_depth_maps):
		# print(target_depth_maps.shape)
		# print(bins[:,1])
		# print(target_depth_maps.min(dim=2)[0].min(dim=2)[0].squeeze(dim=1))
		bin_centers=0.5*(bins[:, 1:]+bins[:, :-1])
		losses=[]
		for i in range(target_depth_maps.shape[0]):
			gt=target_depth_maps[i][target_depth_maps[i]>0.01]
			try:
				max_loss=(bin_centers[i,-1]-gt.max()).abs()
				min_loss=(bin_centers[i, 0]-gt.min()).abs()
			except:
				# plt.imshow(target_depth_maps[i].squeeze().cpu())
				# plt.show()
				pass

			loss=max_loss+min_loss
			losses.append(loss)
		total_loss=sum(losses)

		#print(max_loss.data,min_loss.data)
		return total_loss/target_depth_maps.shape[0]
