import torch
import torch.nn as nn
import numpy as np
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

from thop import profile

from DSTPP.odeModel import LatentODEfunc as st_func
from DSTPP.odeModel import Model_all, c_loss
from  DSTPP.Models import Transformer, Transformer_ST
from functools import partial

import matplotlib.pyplot as plt


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
    parser.add_argument('--seed', type=int, default=2025, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=200, help='')
    parser.add_argument('--loss_type', type=str, default='rmse_tem_spa',
                        choices=['l1', 'l2', 'Euclid', "rmse_tem_spa", "kl_tem_spa"], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices=[1, 2, 3])
    parser.add_argument('--dataset', type=str, default='Earthquake',
                        choices=['Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility',
                                 'HawkesGMM_2d', 'Independent'], help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--latent_dim', type=int, default=64, help='')
    parser.add_argument('--nhidden', type=int, default=64, help='')
    parser.add_argument('--cond_dim', type=int, default=64, help='')
    parser.add_argument('--alph', type=float, default=0.7, help='')
    parser.add_argument('--gama', type=float, default=1, help='Fourier')
    parser.add_argument('--laten', type=str, default="TransFourier", help='rnn,gru,lstm,bilstm,TransFourier')
    parser.add_argument('--c', type=int, default=8, help='Fourier fre')
    parser.add_argument('--lampa', type=float, default=0.7, help='Fourier and Att')
    parser.add_argument('--sample_ts', type=str, default="sin", help='sin,linear,adative')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='dataset sampling rate')

    parser.add_argument('--dec_cond', type=bool, default=False, help='')
    args = parser.parse_args()
    # print(args)
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
    train_data = [[list(i) for i in u if len(list(i)) > 1] for u in train_data]

    train_data = [
        [[i[0], i[0] - u[index - 1][0] + 1e-5 if index > 0 else i[0] + 1e-5] + i[1:] for index, i in enumerate(u)] for u
        in
        train_data]
    sample_rate = opt.sample_rate
    random.seed(42)
    n = int(len(train_data) * sample_rate)
    sampled_data = random.sample(train_data, n)
    print("sampled_data:", len(sampled_data))
    train_data = sampled_data

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
    print()

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

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=False)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=True)
    valloader = get_dataloader(val_data, len(val_data) if len(val_data) <= 1000 else 1000, D=opt.dim, shuffle=True)

    return trainloader, testloader, valloader, (Max, Min)


def Batch2toModel_data(batch, transformer):
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
    return event_time, event_time_origin, event_loc, enc_out, mask


def transData_to_show(data, mask, flag):
    event_time = data[:, :, :1]
    event_loc = data[:, :, 1:]
    event_time_non_mask = []
    event_loc_non_mask = []

    if flag == "real_show":
        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length > 1:
                event_time_non_mask += event_time[index][-length:]
                event_loc_non_mask += event_loc[index][-length:]

    elif flag == "gen_show":
        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length > 1:
                event_time_non_mask += event_time[index][:length]
                event_loc_non_mask += event_loc[index][:length]

    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)
    return event_time_non_mask, event_loc_non_mask


def transData(data, mask, flag):
    event_time = data[:, :, :1]
    event_loc = data[:, :, 1:]
    event_time_non_mask = []
    event_loc_non_mask = []

    if flag == "real":
        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length > 1:
                event_time_non_mask += event_time[index][-length:]
                event_loc_non_mask += event_loc[index][-length:]

    elif flag == "gen":
        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length > 1:
                event_time_non_mask += event_time[index][:length]
                event_loc_non_mask += event_loc[index][:length]

    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)
    return event_time_non_mask, event_loc_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num

if __name__ == "__main__":

    setup_init(opt)
    setproctitle.setproctitle("Model-Training")

    print('dataset:{}'.format(opt.dataset))

    # Specify a directory for logging data

    logdir = "./ModelSave/logs/{}_batch_size_{}_latent_dim_{}_laten_{}".format(opt.dataset, opt.batch_size, opt.latent_dim,opt.laten)
    model_path = './ModelSave/dataset_{}_batch_size_{}_latent_dim_{}_laten_{}_alph_{}_loss_type_{}_gama_{}_sample_ts_{}_lampa_{}/'.format(opt.dataset, opt.batch_size,
                                                                              opt.latent_dim,opt.laten,opt.alph,opt.loss_type,opt.gama,opt.sample_ts,opt.lampa)
    log_path_ = './ModelSave/dataset_{}_batch_size_{}_latent_dim_{}_laten_{}_alph_{}_loss_type_{}_gama_{}_sample_ts_{}_lampa_{}/'.format(opt.dataset, opt.batch_size,
                                                                              opt.latent_dim,opt.laten,opt.alph,opt.loss_type,opt.gama,opt.sample_ts,opt.lampa)

    # model_path = "./ModelSave_0.7s_0.3s/dataset_Earthquake_batch_size_64_latent_dim_64/"
    if not os.path.exists('ModelSave'):
        os.mkdir('ModelSave')
    if not os.path.exists(log_path_):
        os.mkdir(log_path_)
    log_path = os.path.join(log_path_, "evaluation_results.txt")
    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    transformer = Transformer_ST(
        d_model=opt.cond_dim,
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
        num_units=opt.nhidden,
        latent_dim=opt.latent_dim

    ).to(device)
    if opt.laten =="rnn":
        from DSTPP.odeModel import SpatialTemporalRNN as Enc
    elif opt.laten =="gru":
        from DSTPP.odeModel import SpatialTemporalGRU as Enc
    elif opt.laten == "lstm":
        from DSTPP.odeModel import SpatialTemporalLSTM as Enc
    elif opt.laten == "bilstm":
        from DSTPP.odeModel import SpatialTemporalBiLSTM as Enc
    elif opt.laten == "TransFourier":
        from DSTPP.odeModel import SpatialTemporalTransFourier as Enc
    if opt.laten == "TransFourier":
        Enc = Enc(
            latent_dim=opt.latent_dim,
            nhidden=opt.nhidden,
            cond_dim=opt.cond_dim * 3,
            c = opt.c,
            lampa= opt.lampa,
            nbatch=opt.batch_size
        ).to(device)
    else:
        Enc = Enc(
            latent_dim=opt.latent_dim,
            nhidden=opt.nhidden,
            cond_dim=opt.cond_dim * 3,
            nbatch=opt.batch_size
        ).to(device)
    if opt.dec_cond:
        from DSTPP.odeModel import SpatialTemporalDecoderWithCon as Dec
    else:
        from DSTPP.odeModel import SpatialTemporalDecoder as Dec
    Dec = Dec(
        latent_dim=1 + opt.dim,
        nhidden=opt.latent_dim,
        cond_dim=opt.cond_dim * 3,
        nbatch=opt.batch_size
    ).to(device)

    get_loss = c_loss(
        loss_type=opt.loss_type,
        seq_length=1 + opt.dim,
        alph=opt.alph,
        gama = opt.gama
    ).to(device)
    # Model = Model_all(transformer, diffusion)
    Model = Model_all(transformer, LatentODEfunc, Enc, Dec)

    if opt.mode == 'test':
        Model.load_state_dict(torch.load(model_path + 'model_best_spatio.pkl', map_location='cuda:0'), strict=False)
        # model = torch.load(model_path, map_location='cuda:0')
        print('Weight loaded!!')
    # total_params = sum(p.numel() for p in Model.parameters())
    # print(f"Number of parameters: {total_params}")

    trainloader, testloader, valloader, (MAX, MIN) = data_loader(writer)

    warmup_steps = 5

    # training
    optimizer = AdamW(Model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    min_temporal_test = 1e20
    min_spatio_test = 1e20
    for itr in range(1, opt.total_epochs + 1):
        print('epoch:{}'.format(itr))
        if opt.mode == "train":
            with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
                torch.cuda.empty_cache()

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
            total_rmse = total_rmse_s = total_rmse_t = 0.0

            for i, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
                optimizer.zero_grad()
                event_time_non_mask, event_time_origin, event_loc_non_mask, enc_out_non_mask, mask = Batch2toModel_data(
                    batch, Model.transformer)
                x = torch.cat((event_time_non_mask.unsqueeze(2), event_loc_non_mask), dim=-1)  #

                original_lengths = np.count_nonzero(mask[:, :, 0].cpu(), axis=1)
                min_len, max_len = np.min(original_lengths), mask.size(1)  

                out = Model.Enc.forward(x, mask, enc_out_non_mask)

                qz0_mean, qz0_logvar = out[:, :out.size(-1) // 2], out[:, out.size(-1) // 2:]
                epsilon = torch.randn(qz0_mean.size()).to(device=x.device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean 
                if opt.sample_ts == "linear":
                    samp_ts = torch.linspace(0.0, 1.0, steps=max_len, dtype=torch.float32).to(
                        device=z0.device) 
                elif opt.sample_ts == "sin":
                    samp_ts = torch.sin(torch.linspace(0.0, np.pi / 2, steps=max_len, dtype=torch.float32)).to(
                        device=z0.device)
                elif opt.sample_ts == "adative":
                    valid_indices = np.where(original_lengths == max_len)[0]
                    event_time_valid = event_time_origin[valid_indices] #1,270
                    event_time_valid = event_time_valid - event_time_valid[:, 0:1]
                    samp_ts = event_time_valid / (event_time_valid[:, -1:] - event_time_valid[:, 0:1])
                    samp_ts = samp_ts[0].squeeze(dim=0).to(device=z0.device, dtype=torch.float32)
                    try:
                        assert torch.all(samp_ts[:-1] < samp_ts[1:])
                    except AssertionError:
                        for i in range(1, len(samp_ts)):
                            if samp_ts[i] <= samp_ts[i - 1]:
                                samp_ts[i] = samp_ts[i - 1] + 0.0001  
                if opt.adjoint:
                    pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)  
                else:
                    pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)

                pred_x = Model.Dec(pred_z, enc_out_non_mask)  
                real_t, real_s = transData(x, mask, flag="real")
                gen_t, gen_s = transData(pred_x, mask, flag="gen")

                x = torch.cat((real_t, real_s), dim=-1)
                pred_x = torch.cat((gen_t, gen_s), dim=-1)

                # 计算指标
                pred_x = (pred_x) * (
                        torch.tensor([MAX[1:]]).to(device) - torch.tensor([MIN[1:]]).to(device)) + torch.tensor(
                    [MIN[1:]]).to(device)
                x = (x) * (torch.tensor([MAX[1:]]).to(device) - torch.tensor([MIN[1:]]).to(device)) + torch.tensor(
                    [MIN[1:]]).to(device)

                loss, Nlogpx, analytic_kl = get_loss(pred_x, x, qz0_mean, qz0_logvar)
                loss.backward()
                optimizer.step()
                step += 1
                loss_all += loss.item()
                # total_num += event_time_non_mask.shape[0]
                total_num += event_time_non_mask.shape[0]
                # print(loss_all/total_num)
                if Nlogpx != None:
                    vb, vb_temporal, vb_spatial = get_loss.NLL_cal(Nlogpx=Nlogpx, analytic_kl=analytic_kl)
                    vb_all += vb
                    vb_temporal_all += vb_temporal
                    vb_spatial_all += vb_spatial

                writer.add_scalar(tag='Training/loss_step', scalar_value=loss.item(), global_step=itr)
                torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)

            print('Training/loss_epoch_batch', loss_all / len(trainloader))
            # print('Training/NLL_temporal_epoch', vb_temporal_all / total_num)
            # print('Training/NLL_spatial_epoch', vb_spatial_all / total_num)

            writer.add_scalar(tag='Training/loss_epoch', scalar_value=loss_all / len(trainloader),
                              global_step=itr)
            writer.add_scalar(tag='Training/NLL_temporal_epoch', scalar_value=vb_temporal_all / total_num,
                              global_step=itr)
            writer.add_scalar(tag='Training/NLL_spatial_epoch', scalar_value=vb_spatial_all / total_num,
                              global_step=itr)

        if itr % 10 == 0 or opt.mode == "test":
            print('Evaluate!')
            # flops, params = profile(Model.Enc, inputs=(), verbose=False)
            # print(f"Params: {params/1e6:.2f} M,  FLOPs: {flops/1e9:.2f} G")

            with torch.no_grad():
                Model.eval()
                loss_test_all = vb_all = vb_temporal_all = vb_spatial_all = total_num = 0.0
                mae_temporal = rmse_temporal = mae_spatial = rmse_spatial = total_euclidean_distance = 0.0
                rmse_temporal_dstpp = mae_spatial_dstpp = 0.0
                mae_temporal_smash = mae_spatial_smash = 0.0
                total_flops = 0
                total_p = 0
                for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
                    event_time_non_mask, event_time_origin, event_loc_non_mask, enc_out_non_mask, mask = Batch2toModel_data(
                        batch, Model.transformer)

                    x = torch.cat((event_time_non_mask.unsqueeze(2), event_loc_non_mask), dim=-1)  #
                    original_lengths = np.count_nonzero(mask[:, :, 0].cpu(), axis=1)
                    original_lengths_sum = np.sum(original_lengths) - len(original_lengths)
                    min_len, max_len = np.min(original_lengths), mask.size(1)
                    start_time = time.time()

                    out = Model.Enc.forward(x, mask, enc_out_non_mask)

                    qz0_mean, qz0_logvar = out[:, :out.size(-1) // 2], out[:, out.size(-1) // 2:]
                    epsilon = torch.randn(qz0_mean.size()).to(device=x.device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean  # ~，1，32

                    if opt.sample_ts == "linear":
                        samp_ts = torch.linspace(0.0, 1.0, steps=max_len, dtype=torch.float32).to(
                            device=z0.device)  
                    elif opt.sample_ts == "sin":
                        samp_ts = torch.sin(torch.linspace(0.0, np.pi / 2, steps=max_len, dtype=torch.float32)).to(
                            device=z0.device)
                    elif opt.sample_ts == "adative":
                        valid_indices = np.where(original_lengths == max_len)[0]
                        event_time_valid = event_time_origin[valid_indices] #1,270
                        event_time_valid = event_time_valid - event_time_valid[:, 0:1]
                        samp_ts = event_time_valid / (event_time_valid[:, -1:] - event_time_valid[:, 0:1])
                        samp_ts = samp_ts[0].squeeze(dim=0).to(device=z0.device, dtype=torch.float32)
                        try:
                            assert torch.all(samp_ts[:-1] < samp_ts[1:])
                        except AssertionError:
                            for i in range(1, len(samp_ts)):
                                if samp_ts[i] <= samp_ts[i - 1]:
                                    samp_ts[i] = samp_ts[i - 1] + 0.0001  
                    if opt.adjoint:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)
                    else:
                        pred_z = odeint(Model.LatentODEfunc, z0, samp_ts).permute(1, 0, 2)

                    pred_x = Model.Dec(pred_z, enc_out_non_mask)
            
                    real_t, real_s = transData(x, mask, flag="real")
                    gen_t, gen_s = transData(pred_x, mask, flag="gen")
                    real = real_t
                    gen = gen_t
                    real = (real) * (torch.tensor(MAX[1]).to(device) - torch.tensor(MIN[1]).to(device)) + torch.tensor(
                        MIN[1]).to(device)
                    gen = (gen) * (torch.tensor(MAX[1]).to(device) - torch.tensor(MIN[1]).to(device)) + torch.tensor(
                        MIN[1]).to(device)
                    real_temporal = real
                    gen_temporal = gen

                    mae_temporal += torch.mean(torch.abs(real.squeeze(dim=1) - gen.squeeze(dim=1)))  # mae
                    rmse_temporal += torch.sqrt(torch.mean((real.squeeze(dim=1) - gen.squeeze(dim=1)) ** 2))  # rmse

                    # dstpp
                    rmse_temporal_dstpp += ((real - gen) ** 2).sum().item()
                    # smash
                    mae_temporal_smash += torch.abs(real - gen).sum().item()

                    real = real_s
                    real = (real) * (
                            torch.tensor([MAX[2:]]).to(device) - torch.tensor([MIN[2:]]).to(device)) + torch.tensor(
                        [MIN[2:]]).to(device)
                    gen = gen_s
                    gen = (gen) * (
                            torch.tensor([MAX[2:]]).to(device) - torch.tensor([MIN[2:]]).to(device)) + torch.tensor(
                        [MIN[2:]]).to(device)
                    assert real.shape == gen.shape
                    # print(real.shape)
                    real_spatio = real
                    gen_spatio = gen

                    mae_spatial += torch.mean(torch.sum(torch.abs(real - gen), dim=-1) / real.size(-1))
                    rmse_spatial += torch.sqrt(torch.mean(torch.sum((real - gen) ** 2, dim=-1) / real.size(-1)))
                    euclidean_distance = torch.sqrt(torch.sum((real - gen) ** 2, dim=-1))
                    mean_euclidean_distance = torch.mean(euclidean_distance)
                    total_euclidean_distance += mean_euclidean_distance.item()

                    # dstpp
                    mae_spatial_dstpp += torch.sqrt(torch.sum((real - gen) ** 2, dim=-1)).sum().item()  # dstpp
                    # smash
                    mae_spatial_smash += torch.sqrt(torch.sum((real[:, -2:] - gen) ** 2, dim=-1)).sum().item()

                    # rmae_spatial += ((real - gen) ** 2).sum().item()
                    total_num += gen.shape[0]
                    # assert gen.shape[0] == event_time_non_mask.shape[0]

                    x = torch.cat((real_temporal, real_spatio), dim=-1)
                    pred_x = torch.cat((gen_temporal, gen_spatio), dim=-1)

                    loss, Nlogpx, analytic_kl = get_loss(pred_x, x, qz0_mean, qz0_logvar)
                    # loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    loss_test_all += loss.item()

                    if Nlogpx != None:
                        vb, vb_temporal, vb_spatial = get_loss.NLL_cal(Nlogpx=Nlogpx, analytic_kl=analytic_kl)
                        vb_all += vb
                        vb_temporal_all += vb_temporal
                        vb_spatial_all += vb_spatial
                
                # print(f"FLOPs: {(total_flops/len(testloader)) / 1e9:.2f}G")
                # print(f"Params: {(total_p /len(testloader))/ 1e6:.2f}M")

                print("----------------------our-------------------------------")
                print('Evaluation/mae_temporal', mae_temporal / len(testloader))
                print('Evaluation/rmse_temporal', rmse_temporal / len(testloader))
                print('Evaluation/mae_spatio', mae_spatial / len(testloader))
                print('Evaluation/rmse_spatio', rmse_spatial / len(testloader))
                print('Evaluation/euclidean_distance', total_euclidean_distance / len(testloader))
                # print("----------------------dstpp-------------------------------")
                # print('Evaluation/rmse_temporal_dstpp', np.sqrt(rmse_temporal_dstpp / total_num))
                # print('Evaluation/mae_spatio_dstpp', mae_spatial_dstpp / total_num)
                # print("----------------------smash-------------------------------")
                # print('MAE_smash: ', mae_temporal_smash / total_num, mae_spatial_smash / total_num)
                if True:
                    # import logging
                    with open(log_path, 'a') as f:  
                        f.write(f'test/iter: {itr}\n')
                        # f.write(f'test/evaluation_time: {evaluation_time}\n')
                        f.write("----------------------our-------------------------------\n")
                        f.write(f'test/mae_temporal: {mae_temporal / len(testloader)}\n')
                        f.write(f'test/rmse_temporal: {rmse_temporal / len(testloader)}\n')
                        f.write(f'test/mae_spatio: {mae_spatial / len(testloader)}\n')
                        f.write(f'test/rmse_spatio: {rmse_spatial / len(testloader)}\n')
                        f.write(f'test/euclidean_distance: {total_euclidean_distance / len(testloader)}\n')
                        # f.write("----------------------dstpp-------------------------------\n")
                        # f.write(f'test/rmse_temporal_dstpp: {np.sqrt(rmse_temporal_dstpp / total_num)}\n')
                        # f.write(f'test/mae_spatio_dstpp: {mae_spatial_dstpp / total_num}\n')
                        # f.write("----------------------smash-------------------------------\n")
                        # f.write(f'MAE_smash: {mae_temporal_smash / total_num}, {mae_spatial_smash / total_num}\n')

                # print('Training/NLL_spatial_epoch', vb_spatial_all / total_num)

                if opt.mode == 'train':
                    loss_test_all_ = loss_test_all / len(testloader)
                    loss_temporal_all_ = rmse_temporal / len(testloader)
                    loss_spatio_all_ = total_euclidean_distance / len(testloader)

                    if loss_test_all_ > min_loss_test:
                        early_stop += 1
                        if early_stop >= 10:
                            break
                    else:
                        print("min val loss : {} itr : {}".format(loss_test_all_, itr))
                        torch.save(Model.state_dict(), model_path + 'model_best_spatio.pkl')
                        min_loss_test = loss_test_all_
                        early_stop = 0

                    if loss_temporal_all_ > min_temporal_test:
                        early_stop += 1
                        if early_stop >= 10:
                            break
                    else:
                        print("min val temporal : {} itr : {}".format(loss_temporal_all_, itr))
                        torch.save(Model.state_dict(), model_path + 'model_best_temporal.pkl')
                        min_temporal_test = loss_temporal_all_
                        early_stop = 0

                    if loss_spatio_all_ > min_spatio_test:
                        early_stop += 1
                        if early_stop >= 10:
                            break
                    else:
                        print("min val spatio : {} itr : {}".format(loss_spatio_all_, itr))
                        torch.save(Model.state_dict(), model_path + 'model_best_spatio.pkl')
                        min_spatio_test = loss_spatio_all_
                        early_stop = 0

                    writer.add_scalar(tag='Evaluation/mae_temporal', scalar_value=mae_temporal / len(testloader),
                                      global_step=itr)
                    writer.add_scalar(tag='Evaluation/rmse_temporal', scalar_value=rmse_temporal / len(testloader),
                                      global_step=itr)
                    writer.add_scalar(tag='Evaluation/mae_spatio', scalar_value=mae_spatial / len(testloader),
                                      global_step=itr)
                    writer.add_scalar(tag='Evaluation/rmae_spatio', scalar_value=rmse_spatial / len(testloader),
                                      global_step=itr)
                    writer.add_scalar(tag='Evaluation/total_euclidean_distance',
                                      scalar_value=total_euclidean_distance / len(testloader),
                                      global_step=itr)

                    # torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))
                    torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))
                    print('Model Saved to {}'.format(model_path + 'model_{}.pkl').format(itr))
                if opt.mode == 'test':
                    break  # 跳出

"""
python ode_dstpp.py --dataset Earthquake --laten Fourier
python ode_dstpp.py --dataset COVID19 --laten Fourier
"""