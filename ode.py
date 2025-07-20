import torch
import torch.nn as nn
import numpy as np
# from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from torch.optim import AdamW, Adam
import argparse
from scipy.stats import kstest
from DSTPP.Dataset import get_dataloader
import time
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
import os
# from tqdm import tqdm
import random
import json
from tqdm import tqdm

from DSTPP.odeModel import LatentODEfunc as st_func
from DSTPP.odeModel import SpatialTemporalRNN as Enc
from DSTPP.odeModel import SpatialTemporalDecoder as Dec
from DSTPP.odeModel import Model_all,c_loss
from DSTPP import Transformer_ST
from functools import partial

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S", TIME)


def normalization(x, MAX, MIN):
    return (x - MIN) / (MAX - MIN)

def renormalization(y, MAX, MIN):
    return y * (MAX - MIN) + MIN


def get_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices=[1, 2, 3])
    parser.add_argument('--dataset', type=str, default='Earthquake',
                        choices=['Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility',
                                 'HawkesGMM_2d', 'Independent'], help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=100, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')

    parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim = 1

if opt.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)


def data_loader(writer):
    f = open('dataset/{}/data_train.pkl'.format(opt.dataset), 'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in
                  train_data]

    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in
                val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in
                 test_data]

    data_all = train_data + test_data + val_data

    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] > 0

    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, len(val_data) if len(val_data) <= 1000 else 1000, D=opt.dim, shuffle=False)

    return trainloader, testloader, valloader, (Max, Min)


def Batch2toModel1(batch, transformer):
    if opt.dim == 1:
        event_time_origin, event_time, lng= map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)

    if opt.dim == 3:
        event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2), height.unsqueeze(dim=2)), dim=-1)

    event_time = event_time.to(device) #相对时间 64 228
    event_time_origin = event_time_origin.to(device) #绝对时间 64 228
    event_loc = event_loc.to(device) #位置64 228 2
    enc_out, mask = transformer(event_loc, event_time_origin) #64  226 192,   64  226  1

    # event_time_non_mask_ode = []
    # event_loc_non_mask_ode = []
    # enc_out_non_mask_ode = []
    #
    # for index in range(mask.shape[0]):
    #     length = int(sum(mask[index]).item())
    #     if length > 1:
    #         ode_out = [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
    #         enc_out_non_mask_ode.append(ode_out)
    #         ode_time = [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
    #         event_time_non_mask_ode.append(ode_time)
    #         ode_loc = [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]
    #         event_loc_non_mask_ode.append(ode_loc)
    # event_time_non_mask_ode = event_time_non_mask_ode
    # event_loc_non_mask_ode = event_loc_non_mask_ode
    # enc_out_non_mask_ode = enc_out_non_mask_ode

    return event_time,event_time_origin,event_loc,enc_out, mask


def Batch2toModel(batch, transformer):
    if opt.dim == 1:
        event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)

    if opt.dim == 3:
        event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2), height.unsqueeze(dim=2)), dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)

    enc_out, mask = transformer(event_loc, event_time_origin)


    enc_out_non_mask = []
    event_time_non_mask = []
    event_loc_non_mask = []
    event_time_non_mask_ode = []
    event_loc_non_mask_ode = []
    enc_out_non_mask_ode = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        if length > 1:
            ode_out = [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]
            enc_out_non_mask_ode.append(ode_out)
            ode_time=[i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_time_non_mask_ode.append(ode_time)
            ode_loc = [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]
            event_loc_non_mask_ode.append(ode_loc)
    event_time_non_mask_ode = event_time_non_mask_ode
    event_loc_non_mask_ode = event_loc_non_mask_ode
    enc_out_non_mask_ode = enc_out_non_mask_ode


    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)

    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num



if __name__ == "__main__":

    setup_init(opt)
    setproctitle.setproctitle("Model-Training")

    print('dataset:{}'.format(opt.dataset))

    # Specify a directory for logging data
    logdir = "./logs/{}_timesteps_{}".format(opt.dataset, opt.timesteps)
    model_path = './ModelSave_0.3t_0.7s/dataset_{}_timesteps_{}/'.format(opt.dataset, opt.timesteps)

    if not os.path.exists('ModelSave_0.3t_0.7s'):
        os.mkdir('ModelSave_0.3t_0.7s')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    # model = ST_Diffusion(
    #     n_steps=opt.timesteps,
    #     dim=1 + opt.dim,
    #     condition=True,
    #     cond_dim=64
    # ).to(device)
    #
    # diffusion = GaussianDiffusion_ST(
    #     model,
    #     loss_type=opt.loss_type,
    #     seq_length=1 + opt.dim,
    #     timesteps=opt.timesteps,
    #     sampling_timesteps=opt.samplingsteps,
    #     objective=opt.objective,
    #     beta_schedule=opt.beta_schedule
    # ).to(device)

    transformer = Transformer_ST(
        d_model=64,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim=opt.dim,
        CosSin=True
    ).to(device)

    LatentODEfunc = st_func(
        dim=1 + opt.dim,
        num_units=64,
        self_condition=False,
        condition=True,
        cond_dim=64,
        cond=None,
        samp_ts = None,
        max_seq_tr=None
    ).to(device)

    Enc = Enc(
        latent_dim=64,
        nhidden=64,
        cond_dim=64 * 3,
        nbatch = opt.batch_size
    ).to(device)

    Dec = Dec(
        latent_dim=1 + opt.dim,
        nhidden=64,
        cond_dim=64 * 3,
        nbatch = opt.batch_size
    ).to(device)
    get_loss = c_loss(
        loss_type=opt.loss_type,
        seq_length=1+opt.dim
    ).to(device)
    # Model = Model_all(transformer, diffusion)
    Model = Model_all(transformer,LatentODEfunc,Enc,Dec)

    trainloader, testloader, valloader, (MAX, MIN) = data_loader(writer)

    warmup_steps = 5

    # training
    optimizer = AdamW(Model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for itr in range(1,opt.total_epochs+1):

        print('epoch:{}'.format(itr))

        if itr % 10 ==1 :
            print('Evaluate!')
            with torch.no_grad():
                Model.eval()
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                rmse,mse,mse_t,rmse_t,mse_s,rmse_s = 0.0, 0.0, 0.0, 0.0, 0.0,0.0

                for i, batch in tqdm(enumerate(valloader), total=len(valloader)):
                    event_time_non_mask, event_time_origin, event_loc_non_mask, enc_out_non_mask, mask = Batch2toModel1(
                        batch, Model.transformer)
                    event_time_non_mask_, event_loc_non_mask_, enc_out_non_mask_ = Batch2toModel(batch, Model.transformer)

                    x = torch.cat((event_time_non_mask.unsqueeze(2), event_loc_non_mask), dim=-1)  #
                    original_lengths = np.count_nonzero(mask[:, :, 0].cpu(), axis=1)
                    original_lengths_sum = np.sum(original_lengths)-len(original_lengths)

                    min_len, max_len = np.min(original_lengths), mask.size(1)
                    # h = Model.Enc.initHidden().to(device)
                    h = torch.zeros(50,64).to(device)

                    for t in reversed(range(min_len)):
                        obs = x[:, t, :]
                        out, h = Model.Enc.forward(obs, enc_out_non_mask, h, t)
                    qz0_mean, qz0_logvar = out[:, :out.size(-1) // 2], out[:, out.size(-1) // 2:]
                    epsilon = torch.randn(qz0_mean.size()).to(device=x.device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean  # ~，1，32
                    samp_ts = torch.linspace(0.0, 1.0, steps=max_len, dtype=torch.float32)
                    samp_ts = samp_ts.to(device=z0.device)
                    Model.LatentODEfunc.cond = enc_out_non_mask
                    Model.LatentODEfunc.samp_ts = event_time_origin.unsqueeze(2)
                    Model.LatentODEfunc.max_seq_tr = max_len
                    if opt.adjoint:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
                    else:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
                    pred_x = Model.Dec(pred_z, enc_out_non_mask)
                    loss, Nlogpx, analytic_kl = get_loss(pred_x * mask, x * mask, qz0_mean, qz0_logvar)

                    vb, vb_temporal, vb_spatial = get_loss.NLL_cal(Nlogpx=Nlogpx, analytic_kl=analytic_kl)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    # real = renormalization(x[:, :, :1].detach().cpu(),MAX[1],MIN[1])
                    # gen = renormalization(pred_x[:, :, :1].detach().cpu(), MAX[1], MIN[1])

                    rel = renormalization(x[:, :, :].detach().cpu(), torch.tensor(MAX[1:]), torch.tensor(MIN[1:]))
                    pre = renormalization(pred_x[:, :, :].detach().cpu(), torch.tensor(MAX[1:]), torch.tensor(MIN[1:]))

                    # real = (x[:, :, :1].detach().cpu() + MIN[1]) * (MAX[1] - MIN[1])
                    # gen = (pred_x[:, :, :1].detach().cpu() + MIN[1]) * (MAX[1] - MIN[1])

                    assert rel.shape == pre.shape
                    # assert real.shape == sampled_seq_temporal_all.shape
                    # mae_temporal += torch.abs(real - gen).mean().item()
                    # rmse_temporal += ((real - gen) ** 2).mean().item()

                    rmse += (torch.sqrt(((pre - rel) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()
                    # # 计算 MSE（均方误差）
                    # # mse += ((pre - rel) ** 2).mean().item()  # 计算整体 MSE
                    # # mse_t += ((pre[:, :, :1] - rel[:, :, :1]) ** 2).mean().item()  # 计算时间步上 MSE
                    rmse_t += (
                        torch.sqrt(((pre[:, :, :1] - rel[:, :, :1]) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()
                    # # 按空间维度（空间维度）计算 MSE
                    mse_s += (((pre[:, :, 1:] - rel[:, :, 1:]) ** 2).sum(dim=-1) / pre.size(-1)).mean().item()
                    rmse_s += (
                        torch.sqrt(((pre[:, :, 1:] - rel[:, :, 1:]) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()


                    # rmse_temporal_mean += ((real-sampled_seq_temporal_all)**2).sum().item()

                    # real = x[:, :, 1:].detach().cpu()
                    # gen = pred_x[:, :, -opt.dim:].detach().cpu()
                    #
                    # # assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    # real = (real + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))
                    #
                    # gen = (gen + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))

                    assert rel.shape == pre.shape
                    # assert real.shape==sampled_seq_spatial_all.shape
                    # mae_spatial += torch.sqrt(torch.mean((real - gen) ** 2, dim=-1)).mean().item()
                    # mae_spatial_mean += torch.sqrt(torch.sum((real-sampled_seq_spatial_all)**2,dim=-1)).sum().item()

                    total_num += pre.shape[0]

                    assert pre.shape[0] == event_time_non_mask.shape[0]

                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 100:
                        break
                else:
                    early_stop = 0
                print("Overall RMSE = ", rmse / len(testloader))
                print("rmse_t = ", rmse_t / len(testloader))
                print("mse_s = ", mse_s / len(testloader))
                print("rmse_s = ", rmse_s / len(testloader))

                # torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))
                #
                # min_loss_test = min(min_loss_test, loss_test_all)
                #
                # print('Evaluation/distance_spatial_t', mae_spatial / total_num)
                # print('Evaluation/rmse_temporal_t', np.sqrt(rmse_temporal / total_num))
                #
                # writer.add_scalar(tag='Evaluation/loss_val', scalar_value=loss_test_all / total_num, global_step=itr)
                #
                # writer.add_scalar(tag='Evaluation/NLL_val', scalar_value=vb_test_all / total_num, global_step=itr)
                # writer.add_scalar(tag='Evaluation/NLL_temporal_val', scalar_value=vb_test_temporal_all / total_num,
                #                   global_step=itr)
                # writer.add_scalar(tag='Evaluation/NLL_spatial_val', scalar_value=vb_test_spatial_all / total_num,
                #                   global_step=itr)
                #
                # writer.add_scalar(tag='Evaluation/mae_temporal_val', scalar_value=mae_temporal / total_num,
                #                   global_step=itr)
                # writer.add_scalar(tag='Evaluation/rmse_temporal_t', scalar_value=np.sqrt(rmse_temporal / total_num),
                #                   global_step=itr)
                # # writer.add_scalar(tag='Evaluation/rmse_temporal_mean_val',scalar_value=np.sqrt(rmse_temporal_mean/total_num),global_step=itr)
                #
                # writer.add_scalar(tag='Evaluation/distance_spatial_t', scalar_value=mae_spatial / total_num,
                #                   global_step=itr)
                # # writer.add_scalar(tag='Evaluation/distance_spatial_mean_val',scalar_value=mae_spatial_mean/total_num,global_step=itr)
                #
                # # test set-----------------------------
                '''
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in testloader:
                    event_time_non_mask, event_time_origin, event_loc_non_mask, enc_out_non_mask, mask = Batch2toModel1(
                        batch, Model.transformer)
                    optimizer.zero_grad()
                    x = torch.cat((event_time_non_mask.unsqueeze(2), event_loc_non_mask), dim=-1)  #
                    original_lengths = np.count_nonzero(mask[:, :, 0].cpu(), axis=1)
                    min_len, max_len = np.min(original_lengths), mask.size(1)
                    # h = Model.Enc.initHidden().to(device)

                    h = torch.zeros(50, 64).to(device)
                    for t in reversed(range(min_len)):
                        obs = x[:, t, :]
                        out, h = Model.Enc.forward(obs, enc_out_non_mask, h, t)
                    qz0_mean, qz0_logvar = out[:, :out.size(-1) // 2], out[:, out.size(-1) // 2:]
                    epsilon = torch.randn(qz0_mean.size()).to(device=x.device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean  # ~，1，32
                    samp_ts = torch.linspace(0.0, 1.0, steps=max_len, dtype=torch.float32)
                    samp_ts = samp_ts.to(device=z0.device)
                    Model.LatentODEfunc.cond = enc_out_non_mask
                    Model.LatentODEfunc.samp_ts = event_time_origin.unsqueeze(2)
                    Model.LatentODEfunc.max_seq_tr = max_len
                    if opt.adjoint:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
                    else:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
                    pred_x = Model.Dec(pred_z, enc_out_non_mask)

                    loss, Nlogpx, analytic_kl = get_loss(pred_x * mask, x * mask, qz0_mean, qz0_logvar)
                    total_num += gen.shape[0]

                    vb, vb_temporal, vb_spatial = get_loss.NLL_cal(Nlogpx=Nlogpx, analytic_kl=analytic_kl)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial


                print('Evaluation/NLL_temporal_test', vb_test_temporal_all / total_num)
                print('Evaluation/NLL_spatial_test', vb_test_spatial_all / total_num)

                writer.add_scalar(tag='Evaluation/loss_test', scalar_value=loss_test_all / total_num, global_step=itr)

                writer.add_scalar(tag='Evaluation/NLL_test', scalar_value=vb_test_all / total_num, global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test', scalar_value=vb_test_temporal_all / total_num,
                                  global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test', scalar_value=vb_test_spatial_all / total_num,
                                  global_step=itr)
                '''

        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3 - (1e-3 - 5e-5) * (itr - warmup_steps) / opt.total_epochs
                param_group["lr"] = lr

        writer.add_scalar(tag='Statistics/lr', scalar_value=lr, global_step=itr)

        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        rmse, mse, mse_t, rmse_t, mse_s, rmse_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_rmse=total_rmse_s=total_rmse_t = 0.0

        for i, batch in tqdm(enumerate(trainloader), total=len(trainloader)):

            # event_time_non_mask, event_loc_non_mask, enc_out_non_mask
            event_time_non_mask,event_time_origin,event_loc_non_mask,enc_out_non_mask, mask = Batch2toModel1(batch, Model.transformer)

            # event_time_non_mask 64 226 相对时间
            # event_time_origin 64 226 原始的时间数据，绝对时间
            # event_loc_non_mask 64 226 2 原始的空间数据
            # enc_out_non_mask 64 226 192 通过transformer编码器的，时间，空间，时间+空间
            # mask 64 226 1 map补零的记录位置
            '''
            1.考虑数据长度不一致的问题，差距较大，min=22,max=300
            2.如何编码，输入应该就是 相对时间+原始的空间数据  or  绝对时间+原始的空间数据  or 绝对时间+相对时间+原始的空间数据
            3.odeint的时间应该怎么生成(要求保持严格递增的),主要是范围
            4.
            
            '''
            optimizer.zero_grad()
            x = torch.cat((event_time_non_mask.unsqueeze(2), event_loc_non_mask), dim=-1) #
            original_lengths = np.count_nonzero(mask[:, :, 0].cpu(), axis=1)

            min_len, max_len = np.min(original_lengths),mask.size(1)#获得最小的，最大的数据长度，问题（数据补全的不是数据最长的）

            # h = Model.Enc.initHidden().to(device)

            h = torch.zeros(opt.batch_size, 64).to(device)
            for t in reversed(range(min_len)):
                obs = x[:, t, :] #反向取得，rnn，从min_len开始
                out, h = Model.Enc.forward(obs,enc_out_non_mask, h, t)

            # out, h_out = Model.Enc(x,enc_out_non_mask,min_seq_tr)
            # print(out.shape) #torch.Size([5674, 1, 64])
            # print(h_out.shape)
            qz0_mean, qz0_logvar = out[:,:out.size(-1)//2],  out[:,out.size(-1)//2:]
            epsilon = torch.randn(qz0_mean.size()).to(device=x.device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean #~，1，32
            # z0 = z0 * mask
            # samp_ts =  #~
            # samp_ts = torch.arange(1, x.size(0)+1, dtype=torch.float32)
            # samp_tts = torch.arange(1, 2, dtype=torch.float32)
            # samp_ts = torch.arange(1, x.size(1) + 1, dtype=torch.float32)
            # samp_ts = torch.rand(max_len, dtype=torch.float32)
            # samp_ts, _ = torch.sort(samp_ts)
            # samp_ts[0] = 0.0
            # samp_ts[-1] = 1.0
            #
            samp_ts = torch.linspace(0.0, 1.0, steps=max_len, dtype=torch.float32) #生成时间，范围是[0，1]，大小为mx_len
            #计数器

            samp_ts = samp_ts.to(device=z0.device)
            # print(samp_ts.shape)
            # z0=mean +var*g
            # specific_func = partial(Model.LatentODEfunc, cond=enc_out_non_mask) #额外参数
            # 检查 specific_func 是否是 nn.Module 的实例
            # print(isinstance(specific_func, nn.Module))
            #设置模型的一部i分参数
            Model.LatentODEfunc.cond = enc_out_non_mask
            Model.LatentODEfunc.samp_ts = event_time_origin.unsqueeze(2)
            Model.LatentODEfunc.max_seq_tr = max_len
            if opt.adjoint:
                # print("zz",z0)
                # print(samp_ts.shape)
                # pred_z = odeint(Model.LatentODEfunc, z0, samp_ts)
                pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2) #odeint求解
            else:
                # specific_func = partial(Model.LatentODEfunc, cond=enc_out_non_mask)
                # forward in time and solve ode for reconstructions
                pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
            # print(pred_z.squeeze(0).shape) #samp_ts ~ 1 32
            # print(pred_z)  # samp_ts ~ 1 32
            # print(enc_out_non_mask.shape) #~ 192
            # print(h_out.shape) # torch.Size([1, 5065, 64])
            # pred_z_last = pred_z[-1]
            pred_x= Model.Dec(pred_z,enc_out_non_mask) #解码
            # print(pred_x * mask)
            # print("rmse=",((x * mask - pred_x * mask) ** 2).sum().item())
            pre = pred_x.detach().cpu() * mask.detach().cpu()
            rel = x.detach().cpu() * mask.detach().cpu()

            # print("---------")
            # print("rmse = ", np.sqrt(((pre - rel) ** 2)).sum().item())
            # print("mse_t = ", np.sqrt(((pre[:,:,:1] - rel[:,:,:1]) ** 2)).sum().item())
            # print("mse_s = ", np.sqrt(((pre[:,:,1:] - rel[:,:,1:]) ** 2)).sum().item())
            # rmse = torch.sqrt(((pre - rel) ** 2).mean()).item()
            # mse_t = torch.sqrt(((pre[:, :, :1] - rel[:, :, :1]) ** 2).mean()).item()
            # mse_s = torch.sqrt(((pre[:, :, 1:] - rel[:, :, 1:]) ** 2).mean()).item()
            '''
            #gpt
            mse = torch.mean((pre - rel) ** 2)
            rmse = torch.sqrt(mse)
            total_rmse += rmse.item()  # 累加 RMSE

            mses = torch.mean((pre[:, :, :1] - rel[:, :, :1]) ** 2)
            rmses = torch.sqrt(mses)
            total_rmse_s += rmses.item()  # 累加 RMSE

            mset = torch.mean((pre[:, :, 1:] - rel[:, :, 1:]) ** 2)
            rmset = torch.sqrt(mset)
            total_rmse_t += rmset.item()  # 累加 RMSE
            '''
            ## rmse += (torch.sqrt(((pre - rel) ** 2).sum(dim=-1))/pre.size(-1)).mean().item()  # 计算整体 RMSE
            rmse += (torch.sqrt(((pre - rel) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()
            # # 计算 MSE（均方误差）
            # # mse += ((pre - rel) ** 2).mean().item()  # 计算整体 MSE
            # # mse_t += ((pre[:, :, :1] - rel[:, :, :1]) ** 2).mean().item()  # 计算时间步上 MSE
            rmse_t += (torch.sqrt(((pre[:, :, :1] - rel[:, :, :1]) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()
            # # 按空间维度（空间维度）计算 MSE
            mse_s += (((pre[:, :, 1:] - rel[:, :, 1:]) ** 2).sum(dim=-1) / pre.size(-1)).mean().item()
            rmse_s += (torch.sqrt(((pre[:, :, 1:] - rel[:, :, 1:]) ** 2).sum(dim=-1) / pre.size(-1))).mean().item()

            # print("---------")

            # print(x)
            # pred_x = pred_x * mask
            # print(x * mask)
            # print(pred_x * mask)

            loss, Nlogpx, analytic_kl = get_loss(pred_x * mask, x * mask, qz0_mean, qz0_logvar) #算loss
            # loss,Nlogpx ,analytic_kl = get_loss(pred_x,x,qz0_mean,qz0_logvar)

            # compute loss
            # print(loss)
            # print(Nlogpx)
            # print(analytic_kl)
            # loss = Mo`
            # del.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
            loss.backward()
            loss_all += loss.item()
            # total_num += event_time_non_mask.shape[0]
            total_num = event_time_non_mask.shape[0]
            # print(loss_all/total_num)

            vb, vb_temporal, vb_spatial = get_loss.NLL_cal(Nlogpx=Nlogpx,analytic_kl=analytic_kl)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            writer.add_scalar(tag='Training/loss_step', scalar_value=loss.item(), global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()
            step += 1
            # print
        # print("total rmse = ", total_rmse/ len(trainloader))
        # print("t rmse= ", total_rmse_s/ len(trainloader))
        # print('s rmse = ', total_rmse_t / len(trainloader))
        print("Overall RMSE = ", rmse/ len(trainloader))
        # print("Overall MSE = ", mse/ len(trainloader))
        # print("MSE at first time step = ", mse_t/ len(trainloader))
        print("rmse_t = ", rmse_t/ len(trainloader))
        print("mse_s = ", mse_s/ len(trainloader))
        print("rmse_s = ", rmse_s/ len(trainloader))
        print('Training/loss_epoch', loss_all / len(trainloader))
        '''

        print('Training/loss_epoch', loss_all / total_num)
        print('Training/NLL_epoch', vb_all / total_num)
        print('Training/NLL_temporal_epoch', vb_temporal_all / total_num)
        print('Training/NLL_spatial_epoch', vb_spatial_all / total_num)

        writer.add_scalar(tag='Training/loss_epoch', scalar_value=loss_all / total_num, global_step=itr)
        writer.add_scalar(tag='Training/NLL_epoch', scalar_value=vb_all / total_num, global_step=itr)
        writer.add_scalar(tag='Training/NLL_temporal_epoch', scalar_value=vb_temporal_all / total_num,
                          global_step=itr)
        writer.add_scalar(tag='Training/NLL_spatial_epoch', scalar_value=vb_spatial_all / total_num,
                          global_step=itr)
        '''
        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        # print('Training/loss_epoch', loss_all / total_num)
        # print('Training/NLL_epoch', vb_all / total_num)
        # print('Training/NLL_temporal_epoch', vb_temporal_all / total_num)
        # print('Training/NLL_spatial_epoch', vb_spatial_all / total_num)
        #
        # writer.add_scalar(tag='Training/loss_epoch', scalar_value=loss_all / total_num, global_step=itr)
        # writer.add_scalar(tag='Training/NLL_epoch', scalar_value=vb_all / total_num, global_step=itr)
        # writer.add_scalar(tag='Training/NLL_temporal_epoch', scalar_value=vb_temporal_all / total_num, global_step=itr)
        # writer.add_scalar(tag='Training/NLL_spatial_epoch', scalar_value=vb_spatial_all / total_num, global_step=itr)
