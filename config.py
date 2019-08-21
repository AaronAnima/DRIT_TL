import numpy as np
import tensorlayer as tl


class FLAGS(object):
    def __init__(self):
        ''' For training'''
        # input shapes
        self.ch = 64
        self.concat = False
        self.n_layer = 4 # number of layers in G,D
        self.n_scale = 3 # number of scales in multi-D
        self.show_every_step = 1
        self.n_epoch = 100  # "Epoch to train [25]"
        self.zc_dim = 100  # Dim of noise zc
        self.za_dim = 8  # Dim of noise za / appearance vector
        self.c_shape = [64, 64, 256]  # "Dim of content Tensor"
        self.c_dim = 3  # "Number of image channels. [3]")
        self.img_size_h = 256  # Img height
        self.img_size_w = 256  # Img width
        self.concat = False
        # coefficients
        self.lr = 0.0001
        self.lr_Dc = 0.0001
        self.lambda_content = 10
        self.lambda_corss = 10
        self.lambda_domain = 1
        self.lambda_srecon = 10
        self.lambda_latent = 10
        self.lambda_KL = 0.01
        self.lambda_gen = 1
        self.lambda_ms = 1
        self.KL_batch = 16
        self.share = 1
        # optimization
        self.beta1 = 0.5 # "Momentum term of adam [0.5]")
        self.beta2 = 0.9
        self.batch_size_train = 1 # "The number of batch images [1]")
        # save and eval
        self.dataset = "dog2cat" # "The name of dataset [CIFAR_10, MNIST]")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.eval_dir  = "modeltest"
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        self.eval_step = 50 # Evaluation freq during training
        self.len_dataset = 0
        self.step_num = 100000
        self.save_step = 100
        self.param_dir = 'b' + str(self.batch_size_train) + '_' + 'c' + '_' + str(self.c_shape[0]) + \
                         '_' + str(self.c_shape[1]) + '_' + str(self.c_shape[2]) + '_' + 'share' + '_' +\
                        str(self.share) + '_' + 'za' + '_' + str(self.za_dim) + '_' + str(self.dataset)
        # self.param_dir = 'params'
        ''' For eval '''
        self.eval_epoch_num = 10
        self.multi_latent_batch = 4
        self.eval_print_freq = 5000 #
        self.retrieval_print_freq = 200
        self.eval_sample = 1000 # Query num for mAP matrix
        self.nearest_num = 1000 # nearest obj num for each query
        self.batch_size_eval = 1  # batch size for every eval

flags = FLAGS()

