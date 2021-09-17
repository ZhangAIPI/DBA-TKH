import torch
import random
import numpy as np
import time
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
from PIL import Image
import json
import os
import pandas as pd
from foolbox import PyTorchModel
import torchvision.models as models
import timm
from pretrained_models import incres_v2_ens_model,inception_v3_ens_model
from bit_red.compression import jpeg_compress
from rs import predict

global probability
global is_visited_1d
global selected_h, selected_w
global side_length
global model_name
global num

def get_model_name():
    #return 'resnet18'
    #return 'inception-v3'
    #return 'vgg-16'
    #return 'resnet-101'
    #return 'densenet-121'
    #return 'inception-v3-adv'
    #return 'inc-res-v2-ens'
    #return 'bit-red'
    #return 'fd'
    return 'rs'

def get_dataset_path():
    return "/data/zeliang/val"

def get_max_queries():
    return 1000

def get_threshold():
    return 0

def get_seed():
    return 20

def get_num():
    return 2

def get_dba_max_iter_num_in_2d():
    return 2

def get_dim_num():
    return 5

def get_subspace_factor():
    return 1

def get_side_length():
    return side_length

def get_model(device):
    model_name=get_model_name()
    global side_length
    if model_name=='resnet18':
        side_length=224
        model = models.resnet18(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='inception-v3':
        side_length = 299
        model=models.inception_v3(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='vgg-16':
        side_length=224
        model=models.vgg16(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='resnet-101':
        side_length=224
        model=models.resnet101(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='densenet-121':
        side_length=224
        model=models.densenet121(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='inception-v3-adv':
        side_length=299
        model = timm.create_model('adv_inception_v3', pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name=='inc-res-v2-ens':
        side_length=299
        model=incres_v2_ens_model.model(device)
        return model
    elif model_name=='bit-red' or model_name=='fd':
        side_length = 299
        model = inception_v3_ens_model.model(input_device=device)
        return model
    elif model_name=='rs':
        side_length=224
        model=predict.rs_model(input_device=device)
        return model


def get_label(logit):
    _, predict = torch.max(logit, 1)
    return predict


def get_imagenet_labels():
    #response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    #return eval(response.content)
    with open('imagenet_class_index.json') as json_file:
        imagenet_labels=json.load(json_file)
    return imagenet_labels

def save_results(my_intermediates,selected_paths,name,n,max_length):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path='./results/model_%s_n_%d_seed_%d_queries_%d_threshold_%f/'%(get_model_name(),n,get_seed(),get_max_queries(),get_threshold())
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path+name+'_selected_paths.txt', "w") as f:
        f.write(str(selected_paths))
    numpy_results=np.full((n*2,max_length),np.nan)
    for i,my_intermediate in enumerate(my_intermediates):
        length=my_intermediate.shape[0]
        numpy_results[2*i,:length]=my_intermediate[:length,0]
        numpy_results[2*i+1, :length] = my_intermediate[:length, 1]
    pandas_results=pd.DataFrame(numpy_results)
    pandas_results.to_csv(path+name+'_intermediates.csv')


# path='../../val'
def read_imagenet_data(path: str, n: int, net,device):
    labels_str2int = {}
    imagenet_labels = get_imagenet_labels()
    for key in imagenet_labels:
        labels_str2int[imagenet_labels[key][0]] = int(key)

    image_paths = []
    for x in os.listdir(path):
        for y in os.listdir(os.path.join(path, x)):
            if y.endswith('.JPEG'):
                image_paths.append(os.path.join(os.path.join(path, x), y))
    random.seed(get_seed())
    candidate_image_paths = random.sample(image_paths, k=5 * n)
    images = []
    labels = []
    selected_image_paths = []
    for image_path in candidate_image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((side_length, side_length))
        image = np.asarray(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        label = labels_str2int['n' + str(image_path.split('/n')[1][:8])]
        if get_model_name()=='inc-res-v2-ens' or get_model_name()=='bit-red' or get_model_name()=='fd':
            label+=1
        if get_label(net(torch.from_numpy(image / 255).to(device)[np.newaxis, :, :, :])) == label:
            images.append(image)
            labels.append(label)
            selected_image_paths.append(image_path)

        if len(images) >= n:
            break
    images = np.stack(images)
    labels = np.array(labels)
    images = images / 255
    if get_model_name()=='bit-red':
        images=torch.from_numpy(jpeg_compress(images,0.,1.)).to(device)
    elif get_model_name()=='fd':
        images = torch.from_numpy(jpeg_compress(images, 0., 1.)).to(device)
    else:
        images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device)
    return images, labels, selected_image_paths