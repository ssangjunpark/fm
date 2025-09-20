import sys

sys.dont_write_bytecode = True
sys.path.append('../models')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from external.models.unet import ConditionalUnet1D
from external.models.resnet import get_resnet
from external.models.resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
# from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

obs_horizon = 1
pred_horizon = 16
action_dim = 15
action_horizon = 8
num_epochs = 3001
vision_feature_dim = 1549

dataset_metadata = LeRobotDatasetMetadata(repo_id="ssangjunpark/daros0908_1151")

delta_timestamps = {
        "observation.images.top_camera": [0.0],
        "observation.images.left_camera": [0.0],
        "observation.images.right_camera": [0.0],
        "observation.state" : [0.0],
        "action": [0.0],
}

dataset = LeRobotDataset(repo_id="ssangjunpark/daros0908_1151", delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=6,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
)

vision_encoder_top = get_resnet('resnet18')
vision_encoder_top = replace_bn_with_gn(vision_encoder_top)

vision_encoder_left = get_resnet('resnet18')
vision_encoder_left = replace_bn_with_gn(vision_encoder_left)

vision_encoder_right = get_resnet('resnet18')
vision_encoder_right = replace_bn_with_gn(vision_encoder_right)

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
)
nets = nn.ModuleDict({
    'vision_encoder_top': vision_encoder_top,
    'vision_encoder_left': vision_encoder_left,
    'vision_encoder_right': vision_encoder_right,
    'noise_pred_net': noise_pred_net
}).to(device)

sigma = 0.0

ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)

avg_loss_train_list = []

def train():
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        for data in tqdm(dataloader):
            # breakpoint()
            x_img_top = data['observation.images.top_camera'].unsqueeze(1).to(device) # torch.Size([24, 1, 3, 256, 256])
            x_im_left = data['observation.images.top_camera'].unsqueeze(1).to(device)
            x_img_right = data['observation.images.top_camera'].unsqueeze(1).to(device)
            x_pos = data['observation.state'].to(device) # torch.Size([24, 1, 13])
            x_traj = data['action'].repeat(1, 16, 1).to(device) # torch.Size([24, 1, 15])

            x_traj = x_traj.float() 
            x0 = torch.randn(x_traj.shape, device=device) # torch.Size([24, 1, 15]) 
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj) # torch.Size([24]), torch.Size([24, 1, 15]), torch.Size([24, 1, 15])

            # image_features = nets['vision_encoder'](x_img) # (torch.Size([24, 512])
            # obs_features = torch.cat([image_features, x_pos.squeeze(1)], dim=-1) # torch.Size([24, 525]
            image_features_top_camera = nets['vision_encoder_top'](x_img_top.flatten(end_dim=1))
            image_features_top_camera = image_features_top_camera.reshape(*x_img_top.shape[:2], -1) # torch.Size([24, 1, 512])

            image_features_left_camera= nets['vision_encoder_left'](x_im_left.flatten(end_dim=1))
            image_features_left_camera = image_features_left_camera.reshape(*x_im_left.shape[:2], -1) # torch.Size([24, 1, 512])

            image_features_right_camera = nets['vision_encoder_right'](x_img_right.flatten(end_dim=1))
            image_features_right_camera = image_features_right_camera.reshape(*x_img_right.shape[:2], -1) # torch.Size([24, 1, 512])

            obs_features = torch.cat([image_features_top_camera, image_features_left_camera, image_features_right_camera, x_pos], dim=-1) # torch.Size([24, 1, 525])
            obs_cond = obs_features.flatten(start_dim=1) # torch.Size([24, 525])

            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            ema.step(nets.parameters())

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow')

        if epoch == 3000:
            ema.copy_to(nets.parameters())
            PATH = './checkpoint_t/flow_ema_%05d.pth' % epoch
            torch.save({'vision_encoder': nets.vision_encoder.state_dict(),
                        'noise_pred_net': nets.noise_pred_net.state_dict(),
                        }, PATH)



if __name__ == '__main__':
    train()