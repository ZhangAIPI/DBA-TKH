from QEBA.criteria import TargetClass
import argparse
import torchvision.models as models
from PIL import Image
import numpy as np
import dba_attack_utilsV6
import time
import sys

import torch
import requests, random, os
from QEBA.attack_setting import load_pgen
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
from QEBA.pytorch import PyTorchModel
from QEBA.bapp_custom import BAPP_custom
from dba_attack_utilsV6 import get_model, read_imagenet_data

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def MSE(x1, x2):
    return ((x1 - x2) ** 2).mean()


def get_label(logit):
    predict = np.argmax(logit, axis=1)
    return predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model_discretize', action='store_true')
    parser.add_argument('--attack_discretize', action='store_true')
    parser.add_argument('--atk_level', type=int, default=999)
    args = parser.parse_args()
    TASK = 'imagenet'

    output_folder = 'results_test/'

    imagenet_labels = dba_attack_utilsV6.get_imagenet_labels()

    model=get_model(device)
    fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000,
                          discretize=args.model_discretize)

    ###############################
    print("Get understandable ImageNet labels")

    tgt_images, labels, selected_paths = read_imagenet_data(dba_attack_utilsV6.get_dataset_path(), dba_attack_utilsV6.get_num(),
                                                            get_model(device),device)
    # x_adv_list = torch.zeros_like(tgt_images)
    tgt_images = tgt_images.cpu().numpy()
    tgt_labels = labels.cpu().numpy()
    # tgt_labels = get_label(fmodel.forward(tgt_images))
    init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
    criterion = get_criterion(torch.from_numpy(tgt_labels).to(device))
    src_images = init_attack.run(model, torch.from_numpy(tgt_images).to(device), criterion, early_stop=None)
    src_images = src_images.cpu().numpy()
    src_labels = get_label(fmodel.forward(src_images))

    print(src_images.shape)
    print(tgt_images.shape)
    print("Source Image Label:", src_labels)
    print("Target Image Label:", tgt_labels)
    time_start=time.time()

    for PGEN in ['naive', ]:
        p_gen = load_pgen(TASK, PGEN, args)
        if TASK == 'cifar':
            if PGEN == 'naive':
                ITER = 150
                maxN = 30
                initN = 30
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 150
                maxN = 30
                initN = 30
            elif PGEN.startswith('PCA'):
                ITER = 150
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        elif TASK == 'imagenet' or TASK == 'celeba':
            if PGEN == 'naive':
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN.startswith('PCA'):
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN == 'NNGen':
                ITER = 500
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        # ITER = 20
        print("PGEN: %s" % PGEN)
        adv_list = []
        queries = []
        intermediates = []
        advs_l2 = []
        max_length = 0
        for tgt_image, tgt_label, src_image, src_label in zip(tgt_images, tgt_labels, src_images, src_labels):
            if p_gen is None:
                rho = 1.0
            else:
                rvs = p_gen.generate_ps(src_image, 10, level=999)
                grad_gt = fmodel.gradient_one(src_image, label=src_label)
                rho = p_gen.calc_rho(grad_gt, src_image).item()
            print("rho: %.4f" % rho)
            attack = BAPP_custom(fmodel, criterion=TargetClass(src_label))
            adv = attack(tgt_image, tgt_label, starting_point=src_image, iterations=ITER,
                         stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN,
                         initial_num_evals=initN,
                         internal_dtype=np.float32, rv_generator=p_gen, atk_level=args.atk_level, mask=None,
                         batch_size=16,
                         rho_ref=rho, log_every_n_steps=1, suffix=args.suffix + PGEN, verbose=False)
            # print('\n'.join(['%s:%s' % item for item in adv.__dict__.items()]))
            advs_l2.append(np.linalg.norm(tgt_image - adv.perturbed))
            print(adv.distance.value, advs_l2)
            adv_list.append(adv.perturbed)
            queries.append(adv.queries)
            intermediates.append(adv.intermediate)
            if max_length < adv.intermediate.shape[0]:
                max_length = adv.intermediate.shape[0]
            # with open('BAPP_result/attack_%s_%s_%s.log' % (TASK, PGEN, args.suffix), 'w') as outf:
            # json.dump(attack.logger, outf)

        adv_list = np.array(adv_list)
        adv_list = torch.from_numpy(adv_list).to(device).float()
        queries = np.array(queries)
        intermediates = np.array(intermediates)

        print('QEBA Attack Done')

        print("{:.2f} s to run".format(time.time() - time_start))
        print("Results")

        labels_advs = get_model(device)(adv_list).argmax(1)

        for image_i in range(len(src_images)):
            print("QEBA Adversarial Image {}:".format(image_i))
            label_o = int(labels[image_i])
            label_adv = int(labels_advs[image_i])
            print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
            print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
            print("\t- l2 = {}".format(advs_l2[image_i]))
            # print("\t- l_inf = {}".format(hsja_advs_l_inf[image_i]))
            print("\t- {} queries\n".format(queries[image_i]))

            ###############################
        print("Save Results")

        dba_attack_utilsV6.save_results(intermediates, selected_paths, 'qeba', dba_attack_utilsV6.get_num(), max_length)

        print("our:{}".format(torch.mean(torch.Tensor(advs_l2))))
        print("we test %d pictures, and %d pictures are misclassified!" % (
        dba_attack_utilsV6.get_num(), sum(torch.Tensor(advs_l2))==0))
        '''for image_i, o in enumerate(tgt_images):
            adv_i = np.array(adv_list[image_i].cpu().numpy() * 255).astype(np.uint8)
            img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
            img_adv_i.save(os.path.join(output_folder, "qeba_{}_adversarial.jpg".format(image_i)))'''

# todo: 初始点的生成应该保持一致！