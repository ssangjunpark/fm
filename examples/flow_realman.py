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

# stats = {
#     "observation.state": {
#         "mean": [
#             0.6450900358504618,
#             -0.7566365863550044,
#             -1.9531994279392757,
#             -0.636228704556337,
#             0.9294437836957871,
#             0.40238095617180536,
#             -0.2417583820424925,
#             0.4574256425845039,
#             1.8369025219846666,
#             0.6931115202453135,
#             -0.7588352155993766,
#             -0.8551040061881537,
#             -0.3194374451140716
#         ],
#         "std": [
#             0.013746665990684684,
#             0.8947083635061926,
#             0.23333342921136216,
#             0.316393704604747,
#             0.9590241955650118,
#             0.5256390327101169,
#             0.3837092416757562,
#             0.46628761706609057,
#             0.2356420165570302,
#             0.12353852445827644,
#             0.858561457717603,
#             0.7386663433358946,
#             0.544038013839396
#         ],
#         "min": [
#             0.5829905641022622,
#             -2.0000012140882704,
#             -2.200144076361417,
#             -1.2020034901983574,
#             5.285630235915628e-07,
#             -4.1285939978266144e-05,
#             -1.5655383924931467,
#             -0.48632903159862806,
#             1.3272982419158965,
#             0.3717117656315192,
#             -2.3563873869626804,
#             -1.9888990511303644,
#             -1.4946053480293056
#         ],
#         "max": [
#             0.7236919748675446,
#             0.0011243564880443735,
#             -1.5555997571149072,
#             -0.0017115980269224917,
#             2.500092733199697,
#             1.5705404221271622,
#             1.1933578846346397e-05,
#             1.1496382431532963,
#             2.2008470459966234,
#             0.8072224965133082,
#             0.00021778487748337305,
#             0.0013789871781670667,
#             0.34653744316021606
#         ]
#     },
#     "action": {
#         "mean": [
#             0.6446717456379906,
#             -0.7657628130908906,
#             -1.9580253960446106,
#             -0.634653011990276,
#             0.8869591100572214,
#             0.37807312545714544,
#             -0.2751958570699207,
#             0.4565219046962575,
#             1.842060307539211,
#             0.6915813920158211,
#             -0.7308880203976218,
#             -0.7868339618306812,
#             -0.31450784942248516,
#             0.10673031906753427,
#             -0.010911921302857976
#         ],
#         "std": [
#             0.015251444787637682,
#             0.9049480820394832,
#             0.23450817831765391,
#             0.32122138802103406,
#             0.9508210446404274,
#             0.5049988442631763,
#             0.4463907097952005,
#             0.4742034799336201,
#             0.24279558318767666,
#             0.1265935021611503,
#             0.8595673975128267,
#             0.7408858538508936,
#             0.555657166341685,
#             0.18429556153509055,
#             0.35921902595851757
#         ],
#         "min": [
#             0.5609415191755787,
#             -2.0,
#             -2.2000042098978643,
#             -1.2020014095530074,
#             0.0,
#             -7.909514708942844e-08,
#             -1.5697614516562801,
#             -0.5,
#             1.2674452298884091,
#             0.3591622712551098,
#             -2.4,
#             -2.0,
#             -1.5,
#             -0.542462741099916,
#             -0.9999435168371982
#         ],
#         "max": [
#             0.7354723321922138,
#             0.0002298648734126341,
#             -1.5558640287520298,
#             -0.0017360441008008516,
#             2.500039121230637,
#             1.5701936307405195,
#             4.075713843064229e-06,
#             1.15,
#             2.2006957053613156,
#             0.804657612496324,
#             7.336622863805712e-05,
#             0.0013513474675255924,
#             0.39969337124456317,
#             0.6140361615732242,
#             0.9999952977281527
#         ]
#     },
# }


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
# vision_feature_dim = 1549
vision_feature_dim = 4662

dataset_metadata = LeRobotDatasetMetadata(repo_id="Mcen27/HF_RM_BASE_43")

delta_timestamps = {
        "observation.images.top_camera": [-0.4, -0.2, 0.0],
        "observation.images.left_camera": [-0.4, -0.2, 0.0],
        "observation.images.right_camera": [-0.4, -0.2, 0.0],
        "observation.state" : [-0.4, -0.2, 0.0],
        "action": [0.2 * i for i in range(0,16,1)]
}

dataset = LeRobotDataset(repo_id="Mcen27/HF_RM_BASE_43", delta_timestamps=delta_timestamps)

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

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
)

# noise_pred_net = TransformerForDiffusion(
#     input_dim=action_dim,
#     output_dim=action_dim,
#     horizon=pred_horizon,
#     cond_dim=vision_feature_dim
# )


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
            obs_cond = obs_features.flatten(start_dim=1) # torch.Size([24, 525]) # TODO: THIS IS FOR UNET
            # breakpoint()
            # obs_cond = obs_features #TODO: THIS IS FOR TRANSFORMER
            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond) 
            # vt = nets['noise_pred_net'](xt, timestep, obs_cond)
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