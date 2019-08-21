import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import MeanPool2d, ExpandDims, Tile, UpSampling2d, Elementwise, \
    GlobalMeanPool2d, InstanceNorm2d, Lambda, Input, Dense, DeConv2d, Reshape,\
    Conv2d, Flatten, Concat, GaussianNoise, LayerNorm
from tensorlayer.layers import (SubpixelConv2d, ExpandDims)
from tensorlayer.layers import DeConv2d
from utils import SpectralNormConv2d
from config import flags
from tensorlayer.models import Model
import os

w_init = tf.random_normal_initializer(stddev=0.02)
g_init = tf.random_normal_initializer(1., 0.02)
lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)  # tl.act.lrelu(x, 0.2)



# from ICCV17 Semantic-location-gan by Zyh, encode 16*16
# fully convolutional layers, with spectral norm
def get_D(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim)):
    s = flags.img_size_w # output image size [64]
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    channel = 64
    s32 = s // 64
    nx = Input(shape=x_shape, name='imagein')
    n = Conv2d(channel, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nx)

    for i in range(4):
        n = SpectralNormConv2d(channel * 2, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(n)
        channel = channel * 2
    # 8 * 8
    n = Conv2d(1, (s32, s32), (1, 1), padding='VALID', W_init=w_init)(n) # 8 * 8 -> 1 * 1
    n = Reshape([-1, 1])(n)

    return tl.models.Model(inputs=nx, outputs=n)

# share weights in last layer
def create_base_Ec(x_shape, c):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    channel = c
    ni = Input(x_shape)
    nn_y = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(ni)
    nn_y = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn_y)
    nn_y = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn_y)
    nn_y = InstanceNorm2d(act=None, gamma_init=g_init)(nn_y)
    n_y = Elementwise(tf.add)([ni, nn_y])
    return Model(inputs=ni, outputs=n_y)

def get_Ec(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim),
           y_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    channel = 64

    n_xi = Input(x_shape)
    n_yi = Input(y_shape)

    n_x = Conv2d(channel, (7, 7), (1, 1), act=lrelu, W_init=w_init)(n_xi)

    for i in range(2):
        n_x = Conv2d(channel * 2, (3, 3), (2, 2), W_init=w_init)(n_x)
        n_x = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n_x)
        channel = channel * 2

    for i in range(4):
        # res block
        nn_x = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n_x)
        nn_x = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn_x)
        nn_x = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn_x)
        nn_x = InstanceNorm2d(act=None, gamma_init=g_init)(nn_x)
        n_x = Elementwise(tf.add)([n_x, nn_x])

    channel = 64
    n_y = Conv2d(channel, (7, 7), (1, 1), act=lrelu, W_init=w_init)(n_yi)
    for i in range(2):
        n_y = Conv2d(channel * 2, (3, 3), (2, 2), W_init=w_init)(n_y)
        n_y = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n_y)
        channel = channel * 2

    for i in range(1, 4):
        # res block
        nn_y = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n_y)
        nn_y = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn_y)
        nn_y = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn_y)
        nn_y = InstanceNorm2d(act=None, gamma_init=g_init)(nn_y)
        n_y = Elementwise(tf.add)([n_y, nn_y])

    # share the last res-block
    base_layer = create_base_Ec(n_x.shape, channel).as_layer()
    nn_x = base_layer(n_x)
    nn_y = base_layer(n_y)

    # n = GaussianNoise(is_always=False)(n)

    M = Model(inputs=[n_xi, n_yi], outputs=[nn_x, nn_y], name=name)
    return M


# architecture: VAE encoder
# appearance encoder: input X, output
def get_Ea(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None): # input: (1, 256, 256, 3)
    # ref: DRIT source code (Pytorch Implementation)
    w_init = tf.random_normal_initializer(stddev=0.02)
    channel = 64
    ni = Input(x_shape)
    nn = Conv2d(channel, (7, 7), (1, 1), padding='SAME', W_init=w_init, act=tf.nn.relu)(ni)
    channel *= 2
    nn = Conv2d(channel, (4, 4), (2, 2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
    channel *= 2
    ## Basic Blocks * 4
    for i in range(4):
        nn = Conv2d(channel, (4, 4), (2, 2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
    nn = GlobalMeanPool2d()(nn)
    nn = ExpandDims(1)(nn)
    nn = ExpandDims(1)(nn)
    nn = Conv2d(flags.za_dim, (1, 1), (1, 1), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
    nn = Reshape(shape=[-1, flags.za_dim])(nn)
    M = Model(inputs=ni, outputs=nn, name=name)
    return M


# generator: input a, c, txt, output X', encode 64*64
# generator: input a, c, txt, output X', encode 64*64
# author: zbc
def creat_base_G(n_shape):
    ndf = 256
    def SplitLayer(z):
        res = tf.split(z, num_or_size_splits=flags.n_layer, axis=-1)
        return res
    ni = Input(n_shape)
    nzy = Dense(ndf, W_init=w_init, act=tf.nn.relu)(ni)
    nzy = Dense(ndf, W_init=w_init, act=tf.nn.relu)(nzy)
    nzy = Dense(ndf * flags.n_layer, W_init=w_init)(nzy)  # z: (1, 256 * 4)
    nzy = Lambda(SplitLayer)(nzy)  # len = l024 // flags.n_layer  z: (4, 1, 256)
    return Model(inputs=ni, outputs=nzy, name=None)


def get_G(a_x_shape=(None, flags.za_dim), c_x_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]),
          a_y_shape=(None, flags.za_dim), c_y_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]),
          name=None):
    ndf = 256
    # Model for X
    nax = Input(a_x_shape)
    ncx = Input(c_x_shape)
    nay = Input(a_y_shape)
    ncy = Input(c_y_shape)
    zx = nax

    # base layer for reuse

    base_layer = creat_base_G(nax.shape).as_layer()

    # MLP
    nzx = base_layer(zx)
    # mis-resblock
    nccx = ncx
    for i in range(4):  # change num of total resnet layers from 4 to 3
        nzx_tmp = ExpandDims(1)(nzx[i])
        nzx_tmp = ExpandDims(1)(nzx_tmp)
        nzx_tmp = Tile([1, ncx.shape[1], ncx.shape[2], 1])(nzx_tmp)  # expand
        # res block
        nnx = Conv2d(ndf, (3, 3), (1, 1), W_init=w_init, act=None)(nccx)
        nnx = InstanceNorm2d(act=None, gamma_init=g_init)(nnx)
        nnx = Concat(-1)([nnx, nzx_tmp])
        nnx = Conv2d(ndf * 2, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nnx)
        nnx = Conv2d(ndf, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nnx)

        nnx = Conv2d(ndf, (3, 3), (1, 1), W_init=w_init, act=None)(nnx)
        nnx = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nnx)
        nnx = Concat(-1)([nnx, nzx_tmp])
        nnx = Conv2d(ndf * 2, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nnx)
        nnx = Conv2d(ndf, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nnx)
        nccx = Elementwise(tf.add)([nccx, nnx])

    for i in range(2):
        # nc.shape (1, 54, 54, 256)
        nccx = DeConv2d(ndf // 2, (3, 3), (2, 2), W_init=w_init)(nccx)
        nccx = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nccx)
        ndf = ndf // 2

    ndf = 256
    # Model for Y
    zy = nay
    # MLP
    nzy = base_layer(zy)
    # mis-resblock
    nccy = ncy
    for i in range(4):  # change num of total resnet layers from 4 to 3
        nzy_tmp = ExpandDims(1)(nzy[i])
        nzy_tmp = ExpandDims(1)(nzy_tmp)
        nzy_tmp = Tile([1, ncx.shape[1], ncx.shape[2], 1])(nzy_tmp)  # expand
        # res block
        nny = Conv2d(ndf, (3, 3), (1, 1), W_init=w_init, act=None)(nccy)
        nny = InstanceNorm2d(act=None, gamma_init=g_init)(nny)
        nny = Concat(-1)([nny, nzy_tmp])
        nny = Conv2d(ndf * 2, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nny)
        nny = Conv2d(ndf, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nny)

        nny = Conv2d(ndf, (3, 3), (1, 1), W_init=w_init, act=None)(nny)
        nny = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nny)
        nny = Concat(-1)([nny, nzy_tmp])
        nny = Conv2d(ndf * 2, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nny)
        nny = Conv2d(ndf, (1, 1), (1, 1), padding='VALID', W_init=w_init, act=tf.nn.relu)(nny)
        nccy = Elementwise(tf.add)([nccy, nny])

    for i in range(2):
        # nc.shape (1, 54, 54, 256)
        nccy = DeConv2d(ndf // 2, (3, 3), (2, 2), W_init=w_init)(nccy)
        nccy = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nccy)
        ndf = ndf // 2

    nx = DeConv2d(flags.c_dim, (1, 1), (1, 1), W_init=w_init, act=tf.nn.tanh)(nccx)
    ny = DeConv2d(flags.c_dim, (1, 1), (1, 1), W_init=w_init, act=tf.nn.tanh)(nccy)
    M = Model(inputs=[nax, ncx, nay, ncy], outputs=[nx, ny], name=name)
    return M


def get_G_zc(shape_z=(None, flags.zc_dim), gf_dim=64):
    # reference: DCGAN generator
    output_size = 64
    s16 = output_size // 16
    ni = Input(shape_z)
    nn = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, s16, s16, gf_dim * 8])(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    # nn = DeConv2d(256, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)

    nn = DeConv2d(flags.c_shape[2], (5, 5), (2, 2), W_init=w_init)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)

    nn_ = Conv2d(flags.c_shape[2], (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
    nn_ = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn_)
    nn_ = Conv2d(flags.c_shape[2], (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn_)
    nn_ = InstanceNorm2d(act=None, gamma_init=g_init)(nn_)
    nn = Elementwise(tf.add)([nn, nn_])

    return tl.models.Model(inputs=ni, outputs=nn, name='Generator_zc')


# architecture: CNN
# input: (batch_size_train, 256, 256, 3) output: (batch_size_train, 100)
def get_E_x2zc(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name="Encoder_x2zc"):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    # input: (batch_size_train, 256, 256, 3)
    # output: vector (batch_size_train, za_dim)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ni = Input(x_shape)
    n = Conv2d(64, (7, 7), (1, 1), act=tf.nn.relu, W_init=w_init)(ni)
    n = Conv2d(128, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = GlobalMeanPool2d()(n)
    n = Flatten()(n)
    n = Dense(flags.zc_dim)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: CNN
# input: (batch_size_train, 256, 256, 3) output: (batch_size_train, 100)
def get_E_x2za(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name="Encoder_x2za"):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    # input: (batch_size_train, 256, 256, 3)
    # output: vector (batch_size_train, za_dim)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ni = Input(x_shape)
    n = Conv2d(64, (7, 7), (1, 1), act=tf.nn.relu, W_init=w_init)(ni)
    n = Conv2d(128, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = GlobalMeanPool2d()(n)
    n = Flatten()(n)
    n = Dense(flags.za_dim)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: CNN
# input content tensor: (batch_size_train, 64, 64, 256)
# output: likelihood
def get_D_content(c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2])):
    # reference: DRIT resource code -- Pytorch implementation
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.01)  # tl.act.lrelu(x, 0.01)

    ni = Input(c_shape)
    for i in range(3):
        ni = Conv2d(256, (7, 7), (2, 2), act=None, W_init=w_init, padding='SAME')(ni)
        ni = InstanceNorm2d(act=lrelu, gamma_init=g_init)(ni)
    # 64 * 64 -> 8 * 8
    n = Conv2d(256, (4, 4), (1, 1), act=lrelu, padding='VALID', W_init=w_init)(ni)# 8 * 8 -> 1 * 1
    n = Conv2d(1, (1, 1), (1, 1), padding='VALID', W_init=w_init)(n)
    n = Reshape(shape=[-1, 1])(n)
    return tl.models.Model(inputs=ni, outputs=n, name=None)


if __name__ == '__main__':

    x = tf.zeros([1, 216, 216, 3])
    E_xa = get_Ea(x.shape)
    E_xa.train()
    print("Ea:"+str(count_weights(E_xa)))

    xa, xa_mu, xa_var = E_xa(x)  # (1, 8)
    # # print(a_vec.shape)
    # # print(xa_mu.shape)
    # # print(xa_var.shape)
    # # print(xa.shape)
    #
    D = get_D(x.shape)
    D.train()
    print("D:"+str(count_weights(D)))
    label = D(x)
    print(label.shape)
    E_xc = get_Ec(x.shape)
    E_xc.train()
    xc = E_xc(x)
    print("Ec:"+str(count_weights(E_xc)))
    # # print(xc.shape)
    G = get_G(xa.shape, xc.shape)
    G.train()
    print("G:"+str(count_weights(G)))

    img_x = G([xa, xc])  # (1, 256, 256, 3)
    print(img_x.shape)
    #
    zc = tf.zeros([1, 100])
    G_zc = get_G_zc(zc.shape)
    G_zc.train()
    print("G_zc:"+str(count_weights(G_zc)))
    z_content = G_zc(zc)  # (1, 64, 64, 256)
    print(z_content.shape)
    #
    za = tf.zeros([1, 8])
    G_y = get_G(za.shape, z_content.shape)
    G_y.train()
    print("G_y"+str(count_weights(G_y)))
    # img_y = G_y([za, z_content])  # (1, 256, 256, 3)
    # # print(img_y.shape)
    #
    img_y = tf.zeros([1, 216, 216, 3])
    E_y_zc = get_E_x2zc(img_y.shape)
    E_y_za = get_E_x2za(img_y.shape)
    E_y_zc.train()
    E_y_za.train()
    print("E_y_zc"+str(count_weights(E_y_zc)))
    print("E_y_za"+str(count_weights(E_y_za)))
    # zc_ = E_y_zc(img_y)  # (1, 100)
    # za_ = E_y_za(img_y)  # (1, 8)
    # # print(zc.shape)
    # # print(za.shape)
    #

    # label_x = D_x(img_x)  # (1, 1)
    # label_y = D_y(img_y)  # (1, 1)
    # # print(label_x.shape)
    # # print(label_y.shape)
    #
    D_content = get_D_content(xc.shape)
    D_content.train()
    print("D_c"+str(count_weights(D_content)))
    label_c_x = D_content(xc)  # (1, 1)
    # label_c_zc = D_content(z_content)  # (1, 1)
    print(label_c_x.shape)
    # # print(label_c_zc.shape)

    # y = tf.zeros([1, 256, 256, 3])
    # E_ya = get_E_ya(y.shape)
    # E_ya.train()  # (1, 64)
    # ya = E_ya(y)
    # print(ya.shape)
    # y = tf.zeros([1, 256, 256, 3])
    # E_yc = get_E_yc(y.shape)
    # E_yc.train()  # (1, 64, 64, 256)
    # yc = E_yc(y)
    # print(yc.shape)