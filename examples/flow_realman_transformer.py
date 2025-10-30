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
            0.02534289518551391,
            -2.1706827729360842,
            -0.5629096304024529,
            1.6258425985014564,
            -0.012070194241987324,
            -0.4026936016980247,
            0.16193962602116993,
            2.2243634970191066,
            0.5778220932543984,
            -1.859446903273561,
            0.003941503208423424,
            0.40010345702669836,
            0.1257736367567865,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0011328862366133182
        ],
        "std": [
            0.0011519102377312922,
            0.0006507110306489733,
            3.115483684862666e-05,
            5.022074669912852e-05,
            7.636413925308008e-05,
            8.999197614477418e-05,
            0.001131060400388634,
            0.0011528209502594628,
            5.811690515158436e-05,
            5.487550428341056e-05,
            7.500322239459996e-05,
            3.684510267213327e-05,
            0.09293051115539204,
            0.0,
            0.0,
            0.0,
            0.0,
            0.017778863108293116
        ],
        "min": [
            0.024290399301052094,
            -2.1712860649108885,
            -0.5630242393493653,
            1.6257292659759521,
            -0.012424400216341018,
            -0.4028681481361389,
            0.16085410718917847,
            2.223566303253174,
            0.5777171228408814,
            -1.859576733016968,
            0.0037343000270426275,
            0.3999889154434204,
            -0.072,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.19999998807907104
        ],
        "max": [
            0.027448849201202392,
            -2.169471196746826,
            -0.5628323282241822,
            1.6259735649108886,
            -0.011656599700450897,
            -0.4025540542602539,
            0.16397765545845033,
            2.226218710708618,
            0.5779789287567139,
            -1.85934987449646,
            0.004065849918872118,
            0.40018085985183716,
            0.22,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1853
        ]
    },
    "action": {
        "mean": [
            0.022039350199698694,
            -2.17465392875641,
            -0.5677880685807143,
            1.626305065918103,
            -0.011953250041603533,
            -0.40264128961556567,
            0.15869029760361333,
            2.2215768978117514,
            0.5809977180480044,
            -1.8596813755033634,
            0.004118200024962046,
            0.4004076850891396,
            0.051820190440844155,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.004102186256165412
        ],
        "std": [
            4.8949839644200725e-09,
            1.2266108767543242e-06,
            0.0,
            0.0,
            0.0,
            2.113261014771884e-07,
            0.0,
            6.049210766012799e-07,
            3.5301917017943067e-07,
            1.7249341164216735e-07,
            1.1105313913628292e-09,
            0.0,
            0.03823421374265365,
            0.0,
            0.0,
            0.0,
            0.0,
            0.038363392150702565
        ],
        "min": [
            0.022039350199699402,
            -2.174653928756714,
            -0.5677880685806275,
            1.6263050659179688,
            -0.011953250041604042,
            -0.4026412896156311,
            0.15869029760360717,
            2.2215768978118895,
            0.5809977180480957,
            -1.85968137550354,
            0.004118200024962425,
            0.40040768508911134,
            -0.07999999821186066,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.20000000298023224
        ],
        "max": [
            0.022039350199699402,
            -2.174653928756714,
            -0.5677880685806275,
            1.6263050659179688,
            -0.011953250041604042,
            -0.4026412896156311,
            0.15869029760360717,
            2.2215768978118895,
            0.5809977180480957,
            -1.85968137550354,
            0.004118200024962425,
            0.40040768508911134,
            0.07999999821186066,
            0.0,
            0.0,
            0.0,
            0.0,
            0.20000000298023224
        ]
    },
}

stats = {
    "observation.state": {
        "mean": [
            0.022016120961413272,
            -2.174228033329059,
            -0.5676411407182376,
            1.6259850786043974,
            -0.011932712381846825,
            -0.40255360948721675,
            0.15864591376317488,
            2.22115946862082,
            0.5808422500105738,
            -1.8593074342911122,
            0.004089012763807051,
            0.4002916882410896,
            0.0917083292863469,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.001676605244837354
        ],
        "std": [
            3.772859461829237e-05,
            2.81231301288652e-05,
            1.9375847596666092e-05,
            2.3220130982375812e-05,
            3.338738457032809e-05,
            2.604437437139024e-05,
            3.931015323605387e-05,
            3.6093750009303146e-05,
            3.019085488414636e-05,
            2.41902809095157e-05,
            2.3742020950714844e-05,
            2.2453614876060868e-05,
            0.09307054475657286,
            0.0,
            0.0,
            0.0,
            0.0,
            0.12501300643629867
        ],
        "min": [
            0.021777600449323656,
            -2.1743048542022705,
            -0.5677183069229126,
            1.6259212436676025,
            -0.01216265046596527,
            -0.4026412896156311,
            0.15844599866867065,
            2.221070859527588,
            0.5807883665084839,
            -1.8593847553253173,
            0.003961149966716766,
            0.40023321437835696,
            -0.062,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.3999999761581421
        ],
        "max": [
            0.02223130084872246,
            -2.1741478904724123,
            -0.5675787170410156,
            1.6260607669830323,
            -0.01162170014977455,
            -0.4024842593193054,
            0.15895205359458925,
            2.2213325988769532,
            0.5809453968048096,
            -1.8592276584625245,
            0.004153100095689297,
            0.4003553638458252,
            0.217,
            0.0,
            0.0,
            0.0,
            0.0,
            0.4000000059604645
        ]
    },
    "action": {
        "mean": [
            0.02203935019970165,
            -2.174653928756595,
            -0.5677880685807025,
            1.6263050659180898,
            -0.0119532500416052,
            -0.40264128961558204,
            0.15869029760359224,
            2.221576897811736,
            0.5809977180480481,
            -1.85968137550354,
            0.004118200024962071,
            0.4004076850890884,
            0.039725828837888236,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.002777777819169892
        ],
        "std": [
            0.0,
            4.256623046721727e-07,
            0.0,
            0.0,
            0.0,
            1.7881393432617188e-07,
            3.2100245244082223e-08,
            1.018524616398517e-06,
            2.6976864025341977e-07,
            0.0,
            1.4805813950925255e-09,
            1.5077074129021213e-07,
            0.039999059478174386,
            0.0,
            0.0,
            0.0,
            0.0,
            0.08916522385836148
        ],
        "min": [
            0.022039350199699402,
            -2.174653928756714,
            -0.5677880685806275,
            1.6263050659179688,
            -0.011953250041604042,
            -0.4026412896156311,
            0.15869029760360717,
            2.2215768978118895,
            0.5809977180480957,
            -1.85968137550354,
            0.004118200024962425,
            0.40040768508911134,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.20000000298023224
        ],
        "max": [
            0.022039350199699402,
            -2.174653928756714,
            -0.5677880685806275,
            1.6263050659179688,
            -0.011953250041604042,
            -0.4026412896156311,
            0.15869029760360717,
            2.2215768978118895,
            0.5809977180480957,
            -1.85968137550354,
            0.004118200024962425,
            0.40040768508911134,
            0.07999999821186066,
            0.0,
            0.0,
            0.0,
            0.0,
            0.20000000298023224
        ]
    }
}


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - np.array(stats['min'])) / (np.array(stats['max']) - np.array(stats['min']) + 0.00001)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (np.array(stats['max']) - np.array(stats['min'])) + np.array(stats['min'])
    return data

obs_horizon = 1
pred_horizon = 16
action_dim = 18
action_horizon = 8
num_epochs = 3001
vision_feature_dim = 1554
# vision_feature_dim = 1
# vision_feature_dim = 4647

dataset_metadata = LeRobotDatasetMetadata(repo_id="Mcen27/HF_RM_BASE_SIMPLE_100")

delta_timestamps = {
        "observation.images.top_camera": [-0.4, -0.2, 0.0],
        "observation.images.front_camera": [-0.4, -0.2, 0.0],
        "observation.images.top_camera_depth": [-0.4, -0.2, 0.0],
        "observation.state" : [-0.4, -0.2, 0.0],
        "action": [0.2 * i for i in range(0,16,1)]
}

dataset = LeRobotDataset(repo_id="Mcen27/HF_RM_BASE_SIMPLE_100", delta_timestamps=delta_timestamps)

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

vision_encoder_front = get_resnet('resnet18')
vision_encoder_front = replace_bn_with_gn(vision_encoder_front)

vision_encoder_top_depth = get_resnet('resnet18')
vision_encoder_top_depth = replace_bn_with_gn(vision_encoder_top_depth)

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
    'vision_encoder_front': vision_encoder_front,
    'vision_encoder_top_depth': vision_encoder_top_depth,
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
            # breakpoint()
            x_img_top = data['observation.images.top_camera'].to(device) # torch.Size([24, 3, 3, 256, 256])
            x_im_left = data['observation.images.front_camera'].to(device)# torch.Size([24, 3, 3, 256, 256])
            x_img_right = data['observation.images.top_camera_depth'].to(device)# torch.Size([24, 3, 3, 256, 256])
            x_pos = data['observation.state'] # torch.Size([24, 3, 13])
            x_traj = data['action'] # torch.Size([24, 16, 15])

            x_pos = normalize_data(x_pos, stats['observation.state']).to(device)
            x_traj = normalize_data(x_traj, stats['action']).to(device)


            x_traj = x_traj.float()
            x_pos = x_pos.float()
            x0 = torch.randn(x_traj.shape, device=device) # torch.Size([24, 16, 15]) 
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj) # (torch.Size([24]), torch.Size([24, 16, 15]), torch.Size([24, 16, 15]))

            # image_features = nets['vision_encoder'](x_img) # (torch.Size([24, 512])
            # obs_features = torch.cat([image_features, x_pos.squeeze(1)], dim=-1) # torch.Size([24, 525]
            image_features_top_camera = nets['vision_encoder_top'](x_img_top.flatten(end_dim=1)) # torch.Size([72, 512])
            image_features_top_camera = image_features_top_camera.reshape(*x_img_top.shape[:2], -1) # torch.Size([24, 3, 512])

            image_features_front_camera= nets['vision_encoder_front'](x_im_left.flatten(end_dim=1))# torch.Size([72, 512])
            image_features_front_camera = image_features_front_camera.reshape(*x_im_left.shape[:2], -1) # torch.Size([24, 3, 512])

            image_features_top_depth_camera = nets['vision_encoder_top_depth'](x_img_right.flatten(end_dim=1))# torch.Size([72, 512])
            image_features_top_depth_camera = image_features_top_depth_camera.reshape(*x_img_right.shape[:2], -1)# torch.Size([24, 3, 512])

            obs_features = torch.cat([image_features_top_camera, image_features_front_camera, image_features_top_depth_camera, x_pos], dim=-1) # torch.Size([24, 3, 1549])
            # obs_cond = obs_features.flatten(start_dim=1) # torch.Size([24, 525]) # TODO: THIS IS FOR UNET
            # breakpoint()
            obs_cond = obs_features #TODO: THIS IS FOR TRANSFORMER
            # vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond) 
            vt = nets['noise_pred_net'](xt, timestep, obs_cond)
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
                            'vision_encoder_front': nets.vision_encoder_front.state_dict(),
                            'vision_encoder_top_depth': nets.vision_encoder_top_depth.state_dict(),
                            'noise_pred_net': nets.noise_pred_net.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' :lr_scheduler.state_dict(),
                            }, PATH)

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}")


if __name__ == '__main__':
    train()