#!/bin/bash

# Define arguments
min_preceding_timesteps_for_train=13
n_train_timesteps=160
n_val_timesteps=10

# Run Python script with arguments
python gen_data.py --min_preceding_timesteps_for_train $min_preceding_timesteps_for_train \
                   --n_train_timesteps $n_train_timesteps \
                   --n_val_timesteps $n_val_timesteps

python train_model.py

