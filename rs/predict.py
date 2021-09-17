""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
#import setGPU
from rs.datasets import get_num_classes
from rs.core import Smooth
import torch
from rs.architectures import get_architecture
import dba_attack_utilsV6
import traceback
import eagerpy as ep

class rs_model:
    def __init__(self,input_device):
        self.dataset = 'imagenet'
        self.base_classifier = './rs/checkpoint.pth.tar'
        self.sigma = 0.25
        self.batch = 100
        self.N = 100
        self.alpha = 0.001
        self.device=input_device
        self.bounds = [0, 1]
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.mean = self.mean.unsqueeze(1).unsqueeze(1).expand(3, 224, 224)
        self.mean = torch.reshape(self.mean, (1, 3, 224, 224))
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        self.std = self.std.unsqueeze(1).unsqueeze(1).expand(3, 224, 224)
        self.std = torch.reshape(self.std, (1, 3, 224, 224))

        # load the base classifier
        self.checkpoint = torch.load(self.base_classifier)
        self.base_classifier = get_architecture(self.checkpoint["arch"], self.dataset,self.device)
        self.base_classifier.load_state_dict(self.checkpoint['state_dict'])

        # create the smoothed classifier g
        self.smoothed_classifier = Smooth(self.base_classifier, get_num_classes(self.dataset), self.sigma)

    def __call__(self,x):
        #x = (x - self.mean) / self.std
        if str(type(x))=='<class \'eagerpy.tensor.pytorch.PyTorchTensor\'>':
            out = torch.zeros(dba_attack_utilsV6.get_num(), 1001).to(device=self.device)
            x=torch.from_numpy(x.numpy()).to(self.device)
            for i in range(dba_attack_utilsV6.get_num()):
                prediction = self.smoothed_classifier.predict(x[i:i+1], self.N, self.alpha, self.batch)
                out[i,prediction]=1.
        else:
            out = torch.zeros(x.shape[0], 1001).to(device=self.device)
            for i in range(x.shape[0]):
                prediction = self.smoothed_classifier.predict(x[i:i + 1], self.N, self.alpha, self.batch)
                out[i, prediction] = 1.
        if traceback.extract_stack()[-2][2]!='is_adversarial':
            return out
        else:
            return ep.astensor(out)
        return out