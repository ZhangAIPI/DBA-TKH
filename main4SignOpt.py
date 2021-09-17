import json
import torch
import os
import foolbox as fb
import numpy as np
import itertools
import copy
import pickle
import eagerpy as ep
import torchvision.models as models
import argparse
import time
import requests
import dba_attack_utilsV6
from datetime import datetime
from PIL import Image
from foolbox.utils import samples
from foolbox.distances import l2
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox import PyTorchModel
from benchmark import Sign_Opt
from dba_attack_utilsV6 import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
Save Results
baseline9.217354774475098  our:8.834954261779785


Save Results
baseline:10.149131774902344  our:8.887520790100098
we win 45 pictures
"""


# If SurFree integrate in FoolBox Run:
# from foolbox.attacks import SurFree

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="./results_test/", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=dba_attack_utilsV6.get_num(), help="N images attacks")
    parser.add_argument(
        "--config_path",
        default="config_example.json",
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ###############################
    output_folder = args.output_folder


    ###############################
    print("Load Model")
    fmodel = get_model(device)

    ###############################
    print("Load Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise ValueError("{} doesn't exist.".format(args.config_path))
        config = json.load(open(args.config_path, "r"))
    else:
        config = {"init": {}, "run": {"epsilons": None}}

    ###############################
    print("Get understandable ImageNet labels")
    imagenet_labels = dba_attack_utilsV6.get_imagenet_labels()

    ###############################
    print("Load Data")
    # images, labels = samples(fmodel, dataset="imagenet", batchsize=args.n_images)
    images, labels, selected_paths = dba_attack_utilsV6.read_imagenet_data(dba_attack_utilsV6.get_dataset_path(),
                                                                           args.n_images, fmodel,device)
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    ###############################
    print("Attack !")
    time_start = time.time()

    sign_opt_model = Sign_Opt.my_sign_opt(fmodel, images, labels,input_device=device)
    sign_opt_advs, sign_opt_intermediates, max_length = sign_opt_model.attack()
    print('Sign_OPT Attack Done')
    print("{:.2f} s to run".format(time.time() - time_start))
    print("Results")

    sign_opt_labels_advs = fmodel(sign_opt_advs.float()).argmax(1)
    sign_opt_advs_l2 = l2(images, sign_opt_advs)

    for image_i in range(len(images)):
        print("Adversarial Image {}:".format(image_i))
        label_o = int(labels[image_i])

        print("Sign_OPT Adversarial Image {}:".format(image_i))
        label_o = int(labels[image_i])
        label_adv = int(sign_opt_labels_advs[image_i])
        print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
        print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
        print("\t- l2 = {}".format(sign_opt_advs_l2[image_i]))

        ###############################
    print("Save Results")

    dba_attack_utilsV6.save_results(sign_opt_intermediates, selected_paths, 'sign_opt', args.n_images, max_length)

    print("Sign_OPT:{}".format(torch.mean(sign_opt_advs_l2)))
    print("we test %d pictures, and %d pictures are misclassified!" % (args.n_images, sum(sign_opt_advs_l2 == 0)))

    '''for image_i, o in enumerate(images):
        adv_i = np.array(sign_opt_advs[image_i].cpu().numpy() * 255).astype(np.uint8)
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
        img_adv_i.save(os.path.join(output_folder, "sign_opt_{}_adversarial.jpg".format(image_i)))'''
