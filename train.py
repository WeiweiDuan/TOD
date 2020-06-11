from keras.layers import Lambda, Input, Dense, Merge, Concatenate,Multiply, Add, add, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics

from utils import load_data, data_augmentation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from models import vae, categorical
from keras import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras.datasets import mnist

def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0
        
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


os.environ["CUDA_VISIBLE_DEVICES"]="0"
MAP_PATH = './data/toronto.png'
TARGET_SAMPLE_DIR = "./data/target_samples"
MASK_PATH = './data/parkinglot.png'
SUBSET_PATH = './data/subset'
SHIFT_LIST = [-10,-5,0,5,10] #
ROTATION_ANGLE = []#0,5,10,15,20,340,345,355
IMG_SIZE = 28
EPOCHS = 1000
LEARNING_RATE = 0.0001
VAE_MODEL_PATH = ''
LOG_DIR = './logs'
MODEL_PATH = ''
# batch_size = 75
latent_dim = 32#32, 5
intermediate_dim = 512#128, 512
num_cls = 10
optimizer = Adam(lr=LEARNING_RATE)
# optimizer = RMSprop(lr=LEARNING_RATE)
initializer = 'glorot_normal'#'random_uniform'#
original_dim = IMG_SIZE*IMG_SIZE*1
w_recons, w_kl, w_ce = 28.0*28.0, 1.0, 100.0

def qz_graph(x, y, intermediate_dim=512,latent_dim=32):
    concat = Concatenate(axis=-1)([x, y])
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(concat)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
    z_mean = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z_var = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
    return z_mean, z_var, z

def qy_graph(x, num_cls=10):
    layer1 = Dense(256, activation='relu',kernel_initializer = initializer)(x)#256. 64
    layer2 = Dense(128, activation='relu',kernel_initializer = initializer)(layer1)#128, 32
    qy_logit = Dense(num_cls,kernel_initializer = initializer)(layer2)
    qy = Activation(tf.nn.softmax)(qy_logit)
    return qy_logit, qy

def px_graph(z, intermediate_dim=512, original_dim=40*40*3):
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(z)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
    reconstruction = Dense(original_dim, activation='sigmoid',kernel_initializer = initializer)(layer2)
    return reconstruction

def pzy_graph(y, latent_dim=32):
    h = Dense(16, activation='relu',kernel_initializer = initializer)(y)#128
    h = Dense(8, activation='relu',kernel_initializer = initializer)(h)#256, 64
    zp_mean = Dense(latent_dim,kernel_initializer = initializer)(h)
    zp_var = Dense(latent_dim,kernel_initializer = initializer)(h)
    return zp_mean, zp_var

def loss(x, xp, zm, zv, zm_prior, zv_prior, w_mse, w_kl):
    reconstruction_loss = mse(x, xp)
    reconstruction_loss *= w_mse
    kl_loss = (zv_prior-zv)*0.5 + (K.square(zm-zm_prior) + K.exp(zv)) / (2*K.exp(zv_prior)+1e-10) - 0.5
    kl_loss = K.sum(kl_loss, axis=-1) * w_kl
    return reconstruction_loss + kl_loss

def kl_loss(zm, zv, zm_prior, zv_prior, weight):
    loss = (zv_prior-zv)*0.5 + (np.square(zm-zm_prior) + np.exp(zv)) / 2*np.exp(zv_prior) - 0.5
    loss = np.sum(loss, axis=-1) * weight
    return loss

def mse_loss(x, xp, weight):
    return (np.square(x - xp)).mean(axis=None) * weight

def ce_loss(yp, weight):
    return (yp * np.log(yp / np.array([0.20,0.20,0.20,0.20,0.20]))).mean(axis=None) * weight

x, y = Input(shape=(original_dim,)), Input(shape=(num_cls,))
sub_enc = Model([x,y],qz_graph(x, y, intermediate_dim=intermediate_dim, latent_dim=latent_dim))
z = Input(shape=(latent_dim,))
sub_dec = Model(z, px_graph(z, intermediate_dim=intermediate_dim, original_dim=original_dim))

x_u = Input(shape=(original_dim,), name='x_u')
x_l = Input(shape=(original_dim,), name='x_l')
y0 = Input(shape=(num_cls,), name='y0_inputs')
y1 = Input(shape=(num_cls,), name='y1_inputs')


zm_p0,zv_p0 = pzy_graph(y0, latent_dim=latent_dim)
zm_p1,zv_p1 = pzy_graph(y1, latent_dim=latent_dim)


# zm0, zv0, z0 = qz_graph(x_u, y0, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm0, zv0, z0 = sub_enc([x_u, y0])
zm_l, zv_l, z_l = sub_enc([x_l, y0])
zm1, zv1, z1 = qz_graph(x_u, y1, intermediate_dim=intermediate_dim, latent_dim=latent_dim)


# xp_u0 = px_graph(z0, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u0 = sub_dec(z0)
xp_l = sub_dec(z_l)
xp_u1 = px_graph(z1, intermediate_dim=intermediate_dim, original_dim=original_dim)

qy_logit, qy = qy_graph(x_u)

vae = Model([x_u,x_l,y0,y1], [xp_l,xp_u0,xp_u1,qy,zm_l,zv_l,zm0,zv0,zm1,zv1,zm_p0,zv_p0,zm_p1,zv_p1])


cat_loss = qy * K.log(qy / K.constant(np.array([0.5,0.5])))
cat_loss = K.sum(cat_loss, axis=-1) * w_ce

vae_loss = qy[:,0]*loss(x_u,xp_u0,zm0,zv0,zm_p0,zv_p0,w_recons,w_kl)+\
            qy[:,1]*loss(x_u,xp_u1,zm1,zv1,zm_p1,zv_p1,w_recons,w_kl)+\
            loss(x_l,xp_l,zm_l,zv_l,zm_p0,zv_p0,w_recons,w_kl) + cat_loss

vae.add_loss(vae_loss)

vae.summary()

# load data
x_u, _ = load_data.load_wetland_samples(SUBSET_PATH)
np.random.shuffle(x_u)
x_l, target_name = load_data.load_wetland_samples(TARGET_SAMPLE_DIR)
x_l_aug = data_augmentation.data_aug(x_l, SHIFT_LIST, ROTATION_ANGLE)

np.random.shuffle(x_l_aug)

x_l = np.reshape(x_l, [-1, IMG_SIZE*IMG_SIZE*3])
x_l_aug = np.reshape(x_l_aug, [-1, IMG_SIZE*IMG_SIZE*3])
x_u = np.reshape(x_u, [-1, IMG_SIZE*IMG_SIZE*3])
image_size = x_u.shape[1]
original_dim = image_size

x_u = x_u.astype('float32') / 255
x_l = x_l.astype('float32') / 255
x_l_aug = x_l_aug.astype('float32') / 255

np.random.shuffle(x_l_aug)
x_l_aug = np.vstack((x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug, x_l_aug,\
                    x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,\
                     x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,\
                    x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug))


print('target samples shape: ', x_l_aug.shape)
print('all samples shape: ', x_u.shape)


vae.compile(optimizer=optimizer, loss=None)
checkpoint = ModelCheckpoint('./logs/weights{epoch:08d}.h5',save_weights_only=True, period=100)
# vae.load_weights('./logs/weights00000100.h5')
vae.fit([_x_u,_x_l,np.array([[1,0]]*batch_size),np.array([[0,1]]*batch_size)],\
        epochs=200, batch_size=200, verbose=1, callbacks=[checkpoint])
vae.save_weights('dection.hdf5')
