import tensorflow as tf
import numpy as np
import pickle
import os
import sys
from networks import C_VAE_NET, D_VAE_NET
from m3gan import m3gan
from utils import renormlizer

# prepare data for training
with open(r"vital_sign_24hrs_10kx2x3.pkl", 'rb') as f:
    vital_labs_3D = pickle.load(f)

print(vital_labs_3D.shape)

with open(r"med_interv_24hrs_10kx2x3.pkl", 'rb') as f:
    medical_interv_3D = pickle.load(f)

with open(r'data_statics_10kx2x3.pkl', 'rb') as f:
    statics = pickle.load(f)

continuous_x = vital_labs_3D
discrete_x = medical_interv_3D

# timeseries parameters:

time_steps = continuous_x.shape[1]
c_dim = continuous_x.shape[2]
d_dim = discrete_x.shape[2]
no_gen = continuous_x[0]

# hyper-params for training

batch_size = 100
num_pre_epochs = 50
num_epochs = 10
epoch_ckpt_freq = 10
epoch_loss_freq = 2

# network size 

shared_latent_dim = 4
c_z_size = shared_latent_dim
c_noise_dim = int(c_dim/2)
d_z_size = shared_latent_dim
d_noise_dim = int(d_dim/2)

# network hyperparameters

d_rounds=1
g_rounds=3
v_rounds=1
v_lr_pre=0.0005
v_lr=0.0001  
g_lr=0.0001
d_lr=0.0001

alpha_re = 1
alpha_kl = 0.5
alpha_mt = 0.05
alpha_ct = 0.05
alpha_sm = 1
c_beta_adv, c_beta_fm = 1, 20
d_beta_adv, d_beta_fm = 1, 10

enc_size=128
dec_size=128
enc_layers=3
dec_layers=3
keep_prob=0.9
l2scale=0.001
keep_prob=0.9
l2_scale=0.001

c_vae = C_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                  enc_size=enc_size, dec_size=dec_size, 
                  enc_layers=enc_layers, dec_layers=dec_layers, 
                  keep_prob=keep_prob, l2scale=l2scale)

d_vae = D_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=d_dim, z_dim=d_z_size,
                  enc_size=enc_size, dec_size=dec_size, 
                  enc_layers=enc_layers, dec_layers=dec_layers, 
                  keep_prob=keep_prob, l2scale=l2scale)

# create data directory for saving

checkpoint_dir = os.path.join('data/real/', "data/checkpoint/")  
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# pre-training dual-VAE
tf.reset_default_graph()
run_config = tf.ConfigProto()
with tf.Session(config=run_config) as sess:
    model = m3gan(sess=sess,
                  batch_size=batch_size,
                  time_steps=time_steps,
                  num_pre_epochs=num_pre_epochs,
                  num_epochs=num_epochs,
                  checkpoint_dir=checkpoint_dir,
                  epoch_ckpt_freq=epoch_ckpt_freq,
                  epoch_loss_freq=epoch_loss_freq,
                  # params for c
                  c_dim=c_dim, c_noise_dim=c_noise_dim,
                  c_z_size=c_z_size, c_data_sample=continuous_x,
                  c_vae=c_vae,
                  # params for d
                  d_dim=d_dim, d_noise_dim=d_noise_dim,
                  d_z_size=d_z_size, d_data_sample=discrete_x,
                  d_vae=d_vae,
                  # params for training
                  d_rounds=d_rounds, g_rounds=g_rounds, v_rounds=v_rounds,
                  v_lr_pre=v_lr_pre, v_lr=v_lr, g_lr=g_lr, d_lr=d_lr,
                  alpha_re=alpha_re, alpha_kl=alpha_kl, alpha_mt=alpha_mt, 
                  alpha_ct=alpha_ct, alpha_sm=alpha_sm,
                  c_beta_adv=c_beta_adv, c_beta_fm=c_beta_fm, 
                  d_beta_adv=d_beta_adv, d_beta_fm=d_beta_fm)
    model.build()
    model.train()

print("the END")