import tensorflow as tf
import numpy as np
import os
from Contrastivelosslayer import nt_xent_loss
from utils import ones_target, zeros_target, np_sigmoid, np_rounding

class m3gan(object):
    def __init__(self, sess,
                 # -- shared params:
                 batch_size, time_steps,
                 num_pre_epochs, num_epochs,
                 checkpoint_dir, epoch_ckpt_freq, epoch_loss_freq,
                 # -- params for c
                 c_dim, c_noise_dim,
                 c_z_size, c_data_sample,
                 c_vae,
                 # -- params for d
                 d_dim, d_noise_dim,
                 d_z_size, d_data_sample,
                 d_vae,
                 # -- params for training
                 d_rounds, g_rounds, v_rounds,
                 v_lr_pre, v_lr, g_lr, d_lr,
                 alpha_re, alpha_kl, alpha_mt, 
                 alpha_ct, alpha_sm,
                 c_beta_adv, c_beta_fm, 
                 d_beta_adv, d_beta_fm, 
                 # -- label information
                 conditional=False, num_labels=0, statics_label=None):

        self.sess = sess
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.epoch_ckpt_freq = epoch_ckpt_freq
        self.epoch_loss_freq = epoch_loss_freq
        self.statics_label = statics_label

        # params for continuous
        self.c_dim = c_dim
        self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_rnn_vae_net = c_vae

        # params for discrete
        self.d_dim = d_dim
        self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_rnn_vae_net = d_vae

        # params for training
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.v_rounds = v_rounds

        # params for learning rate
        self.v_lr_pre = v_lr_pre
        self.v_lr = v_lr
        self.g_lr = g_lr
        self.d_lr = d_lr
        
        # params for loss scalar
        self.alpha_re = alpha_re
        self.alpha_kl = alpha_kl
        self.alpha_mt = alpha_mt
        self.alpha_ct = alpha_ct
        self.alpha_sm = alpha_sm
        self.c_beta_adv = c_beta_adv
        self.c_beta_fm = c_beta_fm
        self.d_beta_adv = d_beta_adv
        self.d_beta_fm = d_beta_fm

        # params for label information
        self.num_labels = num_labels
        self.conditional = conditional

    def build(self):
        self.build_tf_graph()
        self.build_loss()
        self.saver = tf.train.Saver()

    def save(self, global_id, model_name=None, checkpoint_dir=None):
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, model_name), global_step=global_id)

    def load(self, model_name=None, checkpoint_dir=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(model_name) + 1:])
        return global_id
    
    def build_tf_graph(self):
        # Step 1: VAE training -------------------------------------------------------------------------------------
        # Pretrain vae for c
        if self.conditional:
            self.real_data_label_pl = tf.placeholder(
                dtype=float, shape=[self.batch_size, self.num_labels], name="real_data_label")

        self.c_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="continuous_real_data")

        if self.conditional:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl, self.real_data_label_pl)
        else:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl)

        # add validation set here -------
        self.c_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="vae_validation_c_data")
        if self.conditional:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl)

        # Pretrain vae for d
        self.d_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="discrete_real_data")
        if self.conditional:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl, self.real_data_label_pl)
        else:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl)

        # add validation set here -------
        self.d_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="vae_validation_d_data")
        if self.conditional:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl)
            
    def build_loss(self):

        #################
        # (1) VAE loss  #
        #################
        alpha_re = self.alpha_re
        alpha_kl = self.alpha_kl
        alpha_mt = self.alpha_mt
        alpha_ct = self.alpha_ct
        alpha_sm = self.alpha_sm

        # 1. VAE loss for c

        self.c_re_loss = tf.losses.mean_squared_error(self.c_real_data_pl, self.c_decoded_output) # reconstruction loss for x(-[0,1]
        c_kl_loss = [0] * self.time_steps # KL divergence
        for t in range(self.time_steps):
            c_kl_loss[t] = 0.5 * (tf.reduce_sum(self.c_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.c_vae_mu[t]), 1) - tf.reduce_sum(self.c_vae_logsigma[t] + 1, 1))
        self.c_kl_loss = tf.reduce_mean(tf.add_n(c_kl_loss))

        # 2. Euclidean distance between latent representations from d and c
        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
        self.vae_matching_loss = tf.losses.mean_squared_error(x_latent_1, x_latent_2)

        # 3. Contrastive loss
        self.vae_contra_loss = nt_xent_loss(tf.reshape(x_latent_1, [x_latent_1.shape[0], -1]),
                                            tf.reshape(x_latent_2, [x_latent_2.shape[0], -1]), self.batch_size)

        # 4. If label: conditional VAE and classification cross entropy loss
        if self.conditional:
            # exclude the label information from the latent vector
            x_latent_1_ = x_latent_1[:, :, :-1]
            x_latent_2_ = x_latent_2[:, :, :-1]
            with tf.variable_scope("Shared_VAE/semantic_classifier"):
                vae_flatten_input = tf.compat.v1.layers.flatten(tf.concat([x_latent_1_, x_latent_2_], axis=-1))
                vae_hidden_layer = tf.layers.dense(vae_flatten_input, units=24, activation=tf.nn.relu)
                vae_logits = tf.layers.dense(vae_hidden_layer, units=4, activation=tf.nn.tanh)
            self.vae_semantics_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.cast(self.real_data_label_pl, dtype=tf.int32)), logits=vae_logits))

        if self.conditional:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss
        # vae validation loss
        self.c_vae_valid_loss = tf.losses.mean_squared_error(self.c_vae_test_data_pl, self.c_vae_test_decoded)

        # VAE loss for d (BCE loss)  
        # self.d_re_loss = tf.losses.mean_squared_error(self.d_real_data_pl, self.d_decoded_output) # reconstruction loss
        self.d_re_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true=self.d_real_data_pl, y_pred=self.d_decoded_output, from_logits=False))

        d_kl_loss = [0] * self.time_steps # KL divergence
        for t in range(self.time_steps):
            d_kl_loss[t] = 0.5 * (tf.reduce_sum(self.d_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.d_vae_mu[t]), 1) - tf.reduce_sum(self.d_vae_logsigma[t] + 1, 1))
        self.d_kl_loss = 0.1 * tf.reduce_mean(tf.add_n(d_kl_loss))

        if self.conditional:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss

        # vae validation loss
        self.d_vae_valid_loss = tf.losses.mean_squared_error(self.d_vae_test_data_pl, self.d_vae_test_decoded)

        #######################
        # Optimizer           #
        #######################
        t_vars = tf.trainable_variables()
        c_vae_vars = [var for var in t_vars if 'Continuous_VAE' in var.name]
        d_vae_vars = [var for var in t_vars if 'Discrete_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]

        # Optimizer for c of vae
        self.c_v_op_pre = tf.train.AdamOptimizer(learning_rate=self.v_lr_pre)\
            .minimize(self.c_vae_loss, var_list=c_vae_vars+s_vae_vars)

        # Optimizer for d of vae
        self.d_v_op_pre = tf.train.AdamOptimizer(learning_rate=self.v_lr_pre)\
            .minimize(self.d_vae_loss, var_list=d_vae_vars+s_vae_vars)

        # Optimizer for c of vae
        self.c_v_op = tf.train.AdamOptimizer(learning_rate=self.v_lr) \
            .minimize(self.c_vae_loss, var_list=c_vae_vars + s_vae_vars)

        # Optimizer for d of vae
        self.d_v_op = tf.train.AdamOptimizer(learning_rate=self.v_lr) \
            .minimize(self.d_vae_loss, var_list=d_vae_vars + s_vae_vars)
        
    def train(self):
        #  prepare training data for c
        continuous_x = self.c_data_sample[: int(0.9 * self.c_data_sample.shape[0]), :, :]
        continuous_x_test = self.c_data_sample[int(0.9 * self.c_data_sample.shape[0]) : , :, :]

        # prepare training data for d
        discrete_x = self.d_data_sample[: int(0.9 * self.d_data_sample.shape[0]), :, :]
        discrete_x_test = self.d_data_sample[int(0.9 * self.d_data_sample.shape[0]):, :, :]

        # prepare training data for label
        if self.conditional:
            label_data = self.statics_label[: int(0.9 * self.d_data_sample.shape[0]), :]

        # num of batches
        data_size = continuous_x.shape[0]
        num_batches = data_size // self.batch_size

        tf.global_variables_initializer().run()

        # pretrain step
        print('start pretraining')
        global_id = 0

        for pre in range(self.num_pre_epochs):

            # prepare data for training dataset (same index)
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            if self.conditional:
                label_data_random = label_data[random_idx]

            # validation data
            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            #print("pretraining epoch %d" % pre)

            c_real_data_lst = []
            c_rec_data_lst = []
            d_real_data_lst = []
            d_rec_data_lst = []

            for b in range(num_batches):

                feed_dict = {}
                # feed d data
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                # feed c data
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                # feed label
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                # Pretrain the discrete and continuous vae loss
                _ = self.sess.run(self.c_v_op_pre, feed_dict=feed_dict)
                '''if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.c_vae_summary, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)'''

                _ = self.sess.run(self.d_v_op_pre, feed_dict=feed_dict)
                '''if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)'''

                global_id += 1

                # Pretrain the continuous and discrete VAE loss
                _, c_loss_value = self.sess.run([self.c_v_op_pre, self.c_vae_loss], feed_dict=feed_dict)
                _, d_loss_value = self.sess.run([self.d_v_op_pre, self.d_vae_loss], feed_dict=feed_dict)
            
                if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                    # real data vs. reconstructed data 
                    real_data, rec_data = self.sess.run([self.c_real_data_pl, self.c_decoded_output], feed_dict=feed_dict)
                    c_real_data_lst.append(real_data)
                    # print("real data is, ", real_data, rec_data)
                    c_rec_data_lst.append(rec_data)

                    # real data vs. reconstructed data (rounding to 0 or 1)
                    real_data, rec_data = self.sess.run([self.d_real_data_pl, self.d_decoded_output], feed_dict=feed_dict)
                    d_real_data_lst.append(real_data)
                    d_rec_data_lst.append(np_rounding( rec_data) )
            # Print the loss function value
            print("Pretraining epoch %d, Continuous VAE loss = %f, Discrete VAE loss = %f" % (pre, c_loss_value, d_loss_value))
        

            # visualize
            if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                # visualise_vae(continuous_x_random, np.vstack(c_rec_data_lst), discrete_x_random, np.vstack(d_rec_data_lst), inx=(pre+1))    ---> not commented in actual code 
                print('finish vae reconstructed data saving in pre-epoch ' + str(pre))

                with tf.device('/gpu:0'):
                    np.savez(r"cuda_test.npz", c_real=np.vstack(c_real_data_lst), c_rec=np.vstack(c_rec_data_lst),
                                          d_real=np.vstack(d_real_data_lst), d_rec=np.vstack(d_rec_data_lst))
        
        # saving the pre-trained model

        save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(global_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save(global_id=global_id - 1, model_name='m3gan', checkpoint_dir=save_path)
        '''reconstructed_continuous_data = self.sess.run(self.c_vae_test_decoded, feed_dict=feed_dict)
        print(reconstructed_continuous_data.shape)'''
        print('finish the pretraining')
