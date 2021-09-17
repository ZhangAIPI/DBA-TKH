import json
import torch
import os
import argparse
import time
import dba_attack_utilsV6
from mine import attack
from dba_attack_utilsV6 import get_model
from foolbox.distances import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    parser.add_argument("--output_folder", "-o", default="results_test/", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=dba_attack_utilsV6.get_num()
                        , help="N images attacks")
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
                                                                           dba_attack_utilsV6.get_num(), fmodel,device)
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    ###############################
    print("Attack !")
    time_start = time.time()

    dba_model = attack.BDAattack(fmodel,input_device=device)

    my_advs, q_list, my_intermediates, max_length = dba_model.attack(images, labels, delta_coefficient=0.2)
    print('DBA Attack Done')
    print("{:.2f} s to run".format(time.time() - time_start))
    print("Results")

    my_labels_advs = fmodel(my_advs).argmax(1)
    my_advs_l2 = l2(images, my_advs)

    for image_i in range(len(images)):
        print("My Adversarial Image {}:".format(image_i))
        label_o = int(labels[image_i])
        label_adv = int(my_labels_advs[image_i])
        print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
        print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
        print("\t- l2 = {}".format(my_advs_l2[image_i]))
        # print("\t- l_inf = {}".format(my_advs_l_inf[image_i]))
        print("\t- {} queries\n".format(q_list[image_i]))

        ###############################
    print("Save Results")

    dba_attack_utilsV6.save_results(my_intermediates, selected_paths, 'dba', args.n_images, max_length)

    print("our:{}".format(torch.mean(my_advs_l2)))
    print("we test %d pictures, and %d pictures are misclassified!" % (args.n_images, sum(my_advs_l2 == 0)))
    '''for image_i, o in enumerate(images):
        adv_i = np.array(my_advs[image_i].cpu().numpy() * 255).astype(np.uint8)
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
        img_adv_i.save(os.path.join(output_folder, "my_{}_adversarial.jpg".format(image_i)))'''
