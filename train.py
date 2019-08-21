import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_Y2X_train
from models import get_D, get_Ec, get_Ea, get_G, get_G_zc, get_E_x2zc, get_E_x2za, get_D_content
import random
import argparse
import math
import scipy.stats as stats
import sys
import tensorflow_probability as tfp
import os
temp_out = sys.stdout

parser = argparse.ArgumentParser()
parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
args = parser.parse_args()

E_x_a = get_Ea([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
E_y_a = get_Ea([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
E_c = get_Ec([None, flags.img_size_h, flags.img_size_w, flags.c_dim],
             [None, flags.img_size_h, flags.img_size_w, flags.c_dim])

G = get_G([None, flags.za_dim], [None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]], [None, flags.za_dim],
          [None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]])

D_x = get_D([None, flags.img_size_h, flags.img_size_h, flags.c_dim])
D_y = get_D([None, flags.img_size_h, flags.img_size_h, flags.c_dim])
D_c = get_D_content([None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]])

E_x_a.train()
E_y_a.train()
E_c.train()
G.train()
D_x.train()
D_y.train()
D_c.train()

lr_share = flags.lr
lr_Dc = flags.lr_Dc
D_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
G_E_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
Dc_optimizer = tf.optimizers.Adam(lr_Dc, beta_1=flags.beta1, beta_2=flags.beta2)

dataset = get_Y2X_train()
len_dataset = flags.len_dataset
n_step_epoch = int(len_dataset // flags.batch_size_train)

weights_G_E = E_x_a.trainable_weights  + E_y_a.trainable_weights + \
              E_c.trainable_weights + G.trainable_weights

weights_D = D_x.trainable_weights + D_y.trainable_weights

weights_Dc = D_c.trainable_weights

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)


def discriminator_loss(real, fake, type='gan', fake_random=None, content=False):
    loss = []
    fake_random_loss = 0

    if content :
        real_loss = tl.cost.sigmoid_cross_entropy(real, tf.ones_like(real))
        fake_loss = tl.cost.sigmoid_cross_entropy(fake, tf.zeros_like(fake))
        loss.append(real_loss + fake_loss)
        # for i in range(n_scale):
        #     if type == 'lsgan' :
        #         real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
        #         fake_loss = tf.reduce_mean(tf.square(fake[i]))
        #
        #     if type =='gan' :
        #         real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
        #         fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))
        #
        #     loss.append(real_loss + fake_loss)

    else :
        real_loss = tl.cost.sigmoid_cross_entropy(real, tf.ones_like(real))
        fake_loss = tl.cost.sigmoid_cross_entropy(fake, tf.zeros_like(fake))
        if fake_random != None:
            fake_random_loss = tl.cost.sigmoid_cross_entropy(fake_random, tf.zeros_like(fake_random))
        loss.append(real_loss * 2 + fake_loss + fake_random_loss)
        # for i in range(n_scale) :
        #     if type == 'lsgan' :
        #         real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
        #         fake_loss = tf.reduce_mean(tf.square(fake[i]))
        #         fake_random_loss = tf.reduce_mean(tf.square(fake_random[i]))
        #
        #     if type == 'gan' :
        #         real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
        #         fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))
        #         fake_random_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_random[i]), logits=fake_random[i]))
        #
        #     loss.append(real_loss * 2 + fake_loss + fake_random_loss)

    return sum(loss)


def generator_loss(fake, type='gan', content=False):
    loss = []
    n_scale = len(fake)

    fake_loss = 0

    if content :
        loss.append(tl.cost.sigmoid_cross_entropy(fake, tf.ones_like(fake) * 0.5))
        # for i in range(n_scale):
        #     if type =='lsgan' :
        #         fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 0.5))
        #
        #     if type == 'gan' :
        #         fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=0.5 * tf.ones_like(fake[i]), logits=fake[i]))
        #
        #     loss.append(fake_loss)
    else :
        loss.append(tl.cost.sigmoid_cross_entropy(fake, tf.ones_like(fake) ))
        # for i in range(n_scale) :
        #     if type == 'lsgan' :
        #         fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))
        #
        #     if type == 'gan' :
        #         fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))
        #
        #     loss.append(fake_loss)

    return sum(loss)


def l2_regularize(x) :
    loss = tl.cost.mean_squared_error(x, tf.ones_like(x))

    return loss


def L1_loss(x, y):
    loss = tl.cost.absolute_difference_error(x, y, is_mean=True)

    return loss


def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss


def train(con=False):
    if con == True:
        E_x_a.load_weights('./{}/{}/E_x_a.npz'.format(flags.checkpoint_dir, flags.param_dir))
        E_c.load_weights('./{}/{}/E_c.npz'.format(flags.checkpoint_dir, flags.param_dir))
        E_y_a.load_weights('./{}/{}/E_y_a.npz'.format(flags.checkpoint_dir, flags.param_dir))
        G.load_weights('./{}/{}/G.npz'.format(flags.checkpoint_dir, flags.param_dir))
        D_x.load_weights('./{}/{}/D_x.npz'.format(flags.checkpoint_dir, flags.param_dir))
        D_y.load_weights('./{}/{}/D_y.npz'.format(flags.checkpoint_dir, flags.param_dir))
        D_c.load_weights('./{}/{}/D_c.npz'.format(flags.checkpoint_dir, flags.param_dir))

    t_s = time.time()
    for step, Y_and_X in enumerate(dataset):
        epoch_num = step // n_step_epoch

        Y_img = Y_and_X[0]  # (1, 256, 256, 3)
        X_img = Y_and_X[1]  # (1, 256, 256, 3)

        with tf.GradientTape(persistent=True) as tape:
            # inference

            z = dist.sample([flags.batch_size_train, flags.za_dim])

            X_app_vec = E_x_a(X_img)  # instead of X_app_vec

            X_cont_vec, Y_cont_vec = E_c([X_img, Y_img])

            Y_app_vec = E_y_a(Y_img)  # instead of X_app_vec

            X_cont_vec_logit = D_c(X_cont_vec)
            Y_cont_vec_logit = D_c(Y_cont_vec)

            fake_X, fake_Y = G([X_app_vec, Y_cont_vec, Y_app_vec, X_cont_vec])

            fake_z_X, fake_z_Y = G([z, X_cont_vec, z, Y_cont_vec])

            real_X_logit = D_x(X_img)
            fake_X_logit = D_x(fake_X)
            real_Y_logit = D_y(Y_img)
            fake_Y_logit = D_y(fake_Y)
            fake_z_X_logit = D_x(fake_z_X)
            fake_z_Y_logit = D_y(fake_z_Y)

            self_recon_X, self_recon_Y = G([X_app_vec, X_cont_vec, Y_app_vec, Y_cont_vec])

            temp_x, temp_y = G([z, X_cont_vec, z, Y_cont_vec])
            fake_X_za = E_x_a(temp_x)[0]
            fake_Y_za = E_y_a(temp_y)[0]

            fake_X_app_vec = E_x_a(fake_X)  # instead of X_app_vec

            fake_X_cont_vec, fake_Y_cont_vec = E_c([fake_X, fake_Y])
            fake_Y_app_vec = E_x_a(fake_Y)  # instead of X_app_vec

            recon_X, recon_Y = G([fake_X_app_vec, fake_Y_cont_vec, fake_Y_app_vec, fake_X_cont_vec])

            # Define loss

            g_adv_loss_X = generator_loss(fake_X_logit) + generator_loss(fake_z_X_logit)
            g_adv_loss_Y = generator_loss(fake_Y_logit) + generator_loss(fake_z_Y_logit)

            g_con_loss_X = generator_loss(X_cont_vec_logit, content=True)
            g_con_loss_Y = generator_loss(Y_cont_vec_logit, content=True)

            g_cyc_loss_X = L1_loss(recon_X, X_img)
            g_cyc_loss_Y = L1_loss(recon_Y, Y_img)

            g_rec_loss_X = L1_loss(self_recon_X, X_img)
            g_rec_loss_Y = L1_loss(self_recon_Y, Y_img)

            g_latent_loss_X = L1_loss(fake_X_za, z)
            g_latent_loss_Y = L1_loss(fake_Y_za, z)

            if flags.concat:
                g_kl_loss_X = kl_loss(mean_X, logvar_X) + l2_regularize(content_a)
                g_kl_loss_Y = kl_loss(mean_Y, logvar_Y) + l2_regularize(content_b)
            else:
                g_kl_loss_X = l2_regularize(X_app_vec) + l2_regularize(X_cont_vec)
                g_kl_loss_Y = l2_regularize(Y_app_vec) + l2_regularize(Y_cont_vec)

            d_adv_loss_X = discriminator_loss(real_X_logit, fake_X_logit, fake_z_X_logit)
            d_adv_loss_Y = discriminator_loss(real_Y_logit, fake_Y_logit, fake_z_Y_logit)

            d_con_loss = discriminator_loss(X_cont_vec_logit, Y_cont_vec_logit, content=True)

            Generator_X_domain_loss = flags.lambda_domain * g_adv_loss_X
            Generator_X_content_loss = flags.lambda_content * g_con_loss_X
            Generator_X_cycle_loss = flags.lambda_corss * g_cyc_loss_X
            Generator_X_recon_loss = flags.lambda_srecon * g_rec_loss_X
            Generator_X_latent_loss = flags.lambda_latent * g_latent_loss_X
            Generator_X_kl_loss = flags.lambda_KL * g_kl_loss_X

            Generator_X_loss = Generator_X_domain_loss + \
                               Generator_X_content_loss + \
                               Generator_X_cycle_loss + \
                               Generator_X_recon_loss + \
                               Generator_X_latent_loss + \
                               Generator_X_kl_loss

            Generator_Y_domain_loss = flags.lambda_domain * g_adv_loss_Y
            Generator_Y_content_loss = flags.lambda_content * g_con_loss_Y
            Generator_Y_cycle_loss = flags.lambda_corss * g_cyc_loss_Y
            Generator_Y_recon_loss = flags.lambda_srecon * g_rec_loss_Y
            Generator_Y_latent_loss = flags.lambda_latent * g_latent_loss_Y
            Generator_Y_kl_loss = flags.lambda_KL * g_kl_loss_Y

            Generator_Y_loss = Generator_Y_domain_loss + \
                               Generator_Y_content_loss + \
                               Generator_Y_cycle_loss + \
                               Generator_Y_recon_loss + \
                               Generator_Y_latent_loss + \
                               Generator_Y_kl_loss

            Discriminator_X_loss = flags.lambda_domain * d_adv_loss_X
            Discriminator_Y_loss = flags.lambda_domain * d_adv_loss_Y
            Discriminator_content_loss = flags.lambda_content * d_con_loss

            Generator_loss = Generator_X_loss + Generator_Y_loss
            Discriminator_loss = Discriminator_X_loss + Discriminator_Y_loss
            Discriminator_content_loss = Discriminator_content_loss


        # Updating D_x D_y and D_c
        grad = tape.gradient(Discriminator_loss, weights_D)
        D_optimizer.apply_gradients(zip(grad, weights_D))

        # Updating all Ea Ec Ez G
        grad = tape.gradient(Discriminator_content_loss, weights_Dc)
        Dc_optimizer.apply_gradients(zip(grad, weights_Dc))

        # Updating all Ea Ec Ez G
        grad = tape.gradient(Generator_loss, weights_G_E)
        G_E_optimizer.apply_gradients(zip(grad, weights_G_E))
        del tape

        # show current state
        if np.mod(step, flags.show_every_step) == 0:
            with open("log.txt", "a+") as f:
                sys.stdout = f
                t_e = time.time()
                print("Epoch:[{}/{}][{}/{}]takes {:.1f} sec, adv_x:{:.3f}, adv_y:{:.3f}, "
                      "adv_cont:{:.3f}, cyc_X:{:.3f}, cyc_Y:{:.3f}, rec_X:{:.3f}, rec_Y:{:.3f}"
                      .format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, t_e - t_s,
                       d_adv_loss_X, d_adv_loss_Y, d_con_loss, g_cyc_loss_X, g_cyc_loss_Y, g_rec_loss_X, g_rec_loss_Y))

                sys.stdout = temp_out
                print("Epoch:[{}/{}][{}/{}]takes {:.1f} sec, adv_x:{:.3f}, adv_y:{:.3f}, "
                      "adv_cont:{:.3f}, cyc_X:{:.3f}, cyc_Y:{:.3f}, rec_X:{:.3f}, rec_Y:{:.3f}"
                      .format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, t_e - t_s,
                       d_adv_loss_X, d_adv_loss_Y, d_con_loss, g_cyc_loss_X, g_cyc_loss_Y, g_rec_loss_X, g_rec_loss_Y))
                t_s = time.time()

        if np.mod(step, flags.save_step) == 0 and step != 0:
            E_x_a.save_weights('{}/{}/E_x_a.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_c.save_weights('{}/{}/E_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_y_a.save_weights('{}/{}/E_y_a.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            G.save_weights('{}/{}/G.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_x.save_weights('{}/{}/D_x.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_y.save_weights('{}/{}/D_y.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_c.save_weights('{}/{}/D_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')

            # G.train()

        if np.mod(step, flags.eval_step) == 0:
            E_c.eval()
            E_x_a.eval()
            E_y_a.eval()
            G.eval()
            X_cont, _ = E_c([X_img, Y_img])
            Y_app = E_y_a(Y_img)
            # X_app = E_x_a(X_img)[0]
            _, sys_img = G([Y_app, X_cont, Y_app, X_cont])
            results = tf.concat([sys_img, Y_img, X_img], axis=0)
            E_c.train()
            E_y_a.train()
            G.train()
            E_x_a.train()
            tl.visualize.save_images(results.numpy(), [flags.batch_size_train, 3],
                                     '{}/{}/train_{:02d}_{:04d}.png'.format(flags.sample_dir, flags.param_dir,
                                                                            step // n_step_epoch, step))

if __name__ == '__main__':
    tl.files.exists_or_mkdir(flags.checkpoint_dir + '/' + flags.param_dir)  # checkpoint path
    tl.files.exists_or_mkdir(flags.sample_dir + '/' + flags.param_dir)  # samples path
    train(con=args.is_continue)
