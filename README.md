# IFSTPP

## Installation

- pytorch = 1.13.1+cu117
- python = 3.8

## Model Training

### Use the following command to train IFSTPP

Earthquake
> python ode_dstpp.py --dataset Earthquake --cuda_id 0 --alph 0.7 --gama 1.0 --c 8 --nhidden 64 --sample_ts sin --laten TransFourier

COVID19
> python ode_dstpp.py --dataset COVID19 --cuda_id 0 --alph 0.7 --gama 1.0 --c 8 --nhidden 64 --sample_ts sin --laten TransFourier

Citibike
> python ode_dstpp.py --dataset Citibike --cuda_id 0 --alph 0.7 --gama 1.0 --c 8 --nhidden 64 --sample_ts sin --laten TransFourier
