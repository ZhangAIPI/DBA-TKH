import tensorflow as tf
from . import inception_v3
import torch
import os
import numpy as np
import eagerpy as ep
import traceback

global device

class model:
    def graph(self,x):
        num_classes = 1001
        self.slim = tf.contrib.slim
        with self.slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        pred = end_points_ens4_adv_v3['Predictions']
        return pred

    def __init__(self,input_device):
        global device
        device=input_device
        self.bounds=[0,1]
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.mean = self.mean.unsqueeze(1).unsqueeze(1).expand(3, 299,299)
        self.mean = torch.reshape(self.mean, (1, 3, 299,299))
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
        self.std = self.std.unsqueeze(1).unsqueeze(1).expand(3, 299, 299)
        self.std = torch.reshape(self.std, (1, 3, 299, 299))
        with tf.Graph().as_default():
            # Prepare graph
            batch_shape = [None, 299, 299, 3]
            self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
            self.pred= self.graph(self.x_input)

            s7 = tf.train.Saver(self.slim.get_model_variables(scope='Ens4AdvInceptionV3'))

            config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)
            s7.restore(self.sess, './pretrained_models/ckpt/ens4_adv_inception_v3_rename.ckpt')
    def __call__(self,input):
        if str(type(input))=='<class \'eagerpy.tensor.pytorch.PyTorchTensor\'>':
            input=torch.from_numpy(input.numpy()).to(device)

        input=(input-self.mean)/self.std
        input=input.permute(0,2,3,1)
        input=input.cpu().numpy().astype(np.float32)
        once_pred=self.sess.run(self.pred,feed_dict={self.x_input:input})
        if traceback.extract_stack()[-2][2]!='is_adversarial':
            return torch.from_numpy(once_pred).to(device)
        else:
            return ep.astensor(torch.from_numpy(once_pred).to(device))