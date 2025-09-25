import sys

sys.dont_write_bytecode = True
sys.path.append('../models')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from external.models.unet import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

from external.models.resnet import get_resnet
from external.models.resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
# from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
import einops
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

stats = {
    "observation.state": {
        "mean": [
            0.6450900358504618,
            -0.7566365863550044,
            -1.9531994279392757,
            -0.636228704556337,
            0.9294437836957871,
            0.40238095617180536,
            -0.2417583820424925,
            0.4574256425845039,
            1.8369025219846666,
            0.6931115202453135,
            -0.7588352155993766,
            -0.8551040061881537,
            -0.3194374451140716
        ],
        "std": [
            0.013746665990684684,
            0.8947083635061926,
            0.23333342921136216,
            0.316393704604747,
            0.9590241955650118,
            0.5256390327101169,
            0.3837092416757562,
            0.46628761706609057,
            0.2356420165570302,
            0.12353852445827644,
            0.858561457717603,
            0.7386663433358946,
            0.544038013839396
        ],
        "min": [
            0.5829905641022622,
            -2.0000012140882704,
            -2.200144076361417,
            -1.2020034901983574,
            5.285630235915628e-07,
            -4.1285939978266144e-05,
            -1.5655383924931467,
            -0.48632903159862806,
            1.3272982419158965,
            0.3717117656315192,
            -2.3563873869626804,
            -1.9888990511303644,
            -1.4946053480293056
        ],
        "max": [
            0.7236919748675446,
            0.0011243564880443735,
            -1.5555997571149072,
            -0.0017115980269224917,
            2.500092733199697,
            1.5705404221271622,
            1.1933578846346397e-05,
            1.1496382431532963,
            2.2008470459966234,
            0.8072224965133082,
            0.00021778487748337305,
            0.0013789871781670667,
            0.34653744316021606
        ]
    },
    "action": {
        "mean": [
            0.6446717456379906,
            -0.7657628130908906,
            -1.9580253960446106,
            -0.634653011990276,
            0.8869591100572214,
            0.37807312545714544,
            -0.2751958570699207,
            0.4565219046962575,
            1.842060307539211,
            0.6915813920158211,
            -0.7308880203976218,
            -0.7868339618306812,
            -0.31450784942248516,
            0.10673031906753427,
            -0.010911921302857976
        ],
        "std": [
            0.015251444787637682,
            0.9049480820394832,
            0.23450817831765391,
            0.32122138802103406,
            0.9508210446404274,
            0.5049988442631763,
            0.4463907097952005,
            0.4742034799336201,
            0.24279558318767666,
            0.1265935021611503,
            0.8595673975128267,
            0.7408858538508936,
            0.555657166341685,
            0.18429556153509055,
            0.35921902595851757
        ],
        "min": [
            0.5609415191755787,
            -2.0,
            -2.2000042098978643,
            -1.2020014095530074,
            0.0,
            -7.909514708942844e-08,
            -1.5697614516562801,
            -0.5,
            1.2674452298884091,
            0.3591622712551098,
            -2.4,
            -2.0,
            -1.5,
            -0.542462741099916,
            -0.9999435168371982
        ],
        "max": [
            0.7354723321922138,
            0.0002298648734126341,
            -1.5558640287520298,
            -0.0017360441008008516,
            2.500039121230637,
            1.5701936307405195,
            4.075713843064229e-06,
            1.15,
            2.2006957053613156,
            0.804657612496324,
            7.336622863805712e-05,
            0.0013513474675255924,
            0.39969337124456317,
            0.6140361615732242,
            0.9999952977281527
        ]
    },
}


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - np.array(stats['min'])) / (np.array(stats['max']) - np.array(stats['min']))
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (np.array(stats['max']) - np.array(stats['min'])) + np.array(stats['min'])
    return data

obs_horizon = 1
pred_horizon = 16
action_dim = 15
action_horizon = 8
num_epochs = 3001
# vision_feature_dim = 1549
vision_feature_dim = 4647

dataset_metadata = LeRobotDatasetMetadata(repo_id="ssangjunpark/daros0909_2005")

delta_timestamps = {
        "observation.images.top_camera": [-0.04, -0.02, 0.0],
        "observation.images.left_camera": [-0.04, -0.02, 0.0],
        "observation.images.right_camera": [-0.04, -0.02, 0.0],
        "observation.state" : [-0.04, -0.02, 0.0],
        "action": [0.02 * i for i in range(0,16,1)]
}

dataset = LeRobotDataset(repo_id="ssangjunpark/daros0909_2005", delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=6,
        batch_size=24,
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

# noise_pred_net = ConditionalUnet1D(
    # input_dim=action_dim,
    # global_cond_dim=vision_feature_dim
# )

noise_pred_net = TransformerForDiffusion(
    input_dim=action_dim,
    output_dim=action_dim,
    horizon=pred_horizon,
    cond_dim=vision_feature_dim
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

if False:
    PATH = '/home/isaac/Documents/Github/fm/checkpoint_t/flow_ema_00001.pth'
    state_dict = torch.load(PATH, map_location='cuda')
    nets.vision_encoder_top.load_state_dict(state_dict['vision_encoder_top'])
    nets.vision_encoder_left.load_state_dict(state_dict['vision_encoder_left'])
    nets.vision_encoder_right.load_state_dict(state_dict['vision_encoder_right'])
    nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
    optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    

FM = ConditionalFlowMatcher(sigma=sigma)

avg_loss_train_list = []

def train():
    count = 0
    dt_start = datetime.now()
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        for data in dataloader:
            # k =einops.rearrange(data['observation.images.top_camera'], "b s n ... -> n (b s) ...")
            
            # x_img_top = data['observation.images.top_camera'].unsqueeze(1).to(device) # torch.Size([24, 1, 3, 256, 256])
            # x_im_left = data['observation.images.top_camera'].unsqueeze(1).to(device)
            # x_img_right = data['observation.images.top_camera'].unsqueeze(1).to(device)
            # breakpoint()
            x_img_top = data['observation.images.top_camera'].to(device) # torch.Size([24, 1, 3, 256, 256])
            x_im_left = data['observation.images.left_camera'].to(device)
            x_img_right = data['observation.images.right_camera'].to(device)
            x_pos = data['observation.state'] # torch.Size([24, 1, 13])
            x_traj = data['action'] # torch.Size([24, 16, 15])

            x_pos = normalize_data(x_pos, stats['observation.state']).to(device)
            x_traj = normalize_data(x_traj, stats['action']).to(device)


            x_traj = x_traj.float()
            x_pos = x_pos.float()
            x0 = torch.randn(x_traj.shape, device=device) # torch.Size([64, 16, 15]) 
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
            # obs_cond = obs_features.flatten(start_dim=1) # torch.Size([24, 525]) # TODO: THIS IS FOR UNET
            # breakpoint()
            obs_cond = obs_features #TODO: THIS IS FOR TRANSFORMER
            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond) 

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            ema.step(nets.parameters())
            count += 1
            print(f"epoch: {epoch:>02}, batch: {count},  loss: {loss:.10f}, time: {datetime.now() - dt_start}")

            if count % 5000 == 0:
                ema.copy_to(nets.parameters())
                PATH = './checkpoint_t/flow_ema_%05d.pth' % count
                torch.save({'vision_encoder_top': nets.vision_encoder_top.state_dict(),
                            'vision_encoder_left': nets.vision_encoder_left.state_dict(),
                            'vision_encoder_right': nets.vision_encoder_right.state_dict(),
                            'noise_pred_net': nets.noise_pred_net.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' :lr_scheduler.state_dict(),
                            }, PATH)

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}")


if __name__ == '__main__':
    train()