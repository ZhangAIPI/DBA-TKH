import time, torch
import numpy as np
from numpy import linalg as LA
import os
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import dba_attack_utilsV6
global device
global queries
MAX_ITER = 10000000
import sys
sys.path.append("..")
from dba_attack_utilsV6 import get_side_length,get_model_name


class OPT_attack(object):
    def __init__(self, model):
        self.model = model
        self.log = torch.ones(MAX_ITER, 2)

    def get_log(self):
        return self.log

    def attack_untargeted(self, x0, y0, init_adv, alpha=0.2, beta=0.001, iterations=100000000, query_limit=100000000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        intermediate=[]
        model = self.model
        if type(x0) is torch.Tensor:
            x0 = x0.cpu().numpy()
        if type(init_adv) is torch.Tensor:
            init_adv=init_adv.cpu().numpy()
        if type(y0) is torch.Tensor:
            y0 = y0.item()
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return torch.tensor(x0).cuda()
        query_count = 0

        '''num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        np.random.seed(0)
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0 + theta) != y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        if g_theta == float('inf'):
            num_directions = 500
            best_theta, g_theta = None, float('inf')
            print("Searching for the initial direction on %d random directions: " % (num_directions))
            timestart = time.time()
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if model.predict_label(x0 + theta) != y0:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f" % g_theta)'''

        timestart = time.time()
        best_theta, g_theta = None, float('inf')
        theta = init_adv - x0
        initial_lbd = LA.norm(theta)
        theta /= initial_lbd  # l2 normalize
        lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()

        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return torch.tensor(x0).cuda()
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (
            g_theta, timeend - timestart, query_count))
        self.log[0][0], self.log[0][1] = g_theta, query_count

        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = query_count
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            dist=torch.norm(torch.from_numpy(g_theta*best_theta).to(device))
            intermediate.append([queries,dist.item()])
            if queries>dba_attack_utilsV6.get_max_queries() or dist<=dba_attack_utilsV6.get_threshold():
                break
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u)
                ttt = theta + beta * u
                ttt /= LA.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd=g2, tol=beta / 500)
                opt_count += count
                gradient += (g1 - g2) / beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0 / q * gradient

            if opt_count > query_limit:
                break

            if (i + 1) % 10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, LA.norm(g2 * theta), opt_count))
                # if g2 > prev_obj-stopping:
                #    break
                prev_obj = g2
            self.log[i + 1][0], self.log[i + 1][1] = g2, opt_count + query_count

            min_theta = theta
            min_g2 = g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2,
                                                                      tol=beta / 500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd=min_g2,
                                                                          tol=beta / 500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2

            # print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
        dist=torch.norm(torch.from_numpy(g_theta* best_theta).to(device))
        intermediate.append([ queries,dist.item()])
        target = model.predict_label(x0 + g_theta * best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (
            g_theta, target, query_count + opt_count, timeend - timestart))

        self.log[i + 1:, 0] = g_theta
        self.log[i + 1:, 1] = opt_count + query_count
        print('l2: %.4f' % (dist))
        return torch.tensor(x0 + g_theta * best_theta, dtype=torch.float).to(device),intermediate

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if model.predict_label(x0 + lbd * theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while model.predict_label(x0 + lbd_hi * theta) == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while model.predict_label(x0 + lbd_lo * theta) != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if model.predict_label(x0 + current_best * theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, TARGETED=False, epsilon=None):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target)
        return adv


class my_model:
    def __init__(self, model):
        self.model = model
        '''self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.mean = self.mean.unsqueeze(1).unsqueeze(1).expand(3, get_side_length(), get_side_length())
        self.mean=torch.reshape(self.mean,(1,3,get_side_length(),get_side_length()))
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
        self.std = self.std.unsqueeze(1).unsqueeze(1).expand(3, get_side_length(), get_side_length())
        self.std=torch.reshape(self.std,(1,3,get_side_length(),get_side_length()))'''

    def predict_label(self, input):
        input=torch.from_numpy(input).to(device).float()
        if input.dim()==3:
            input=torch.reshape(input,(1,input.shape[0],input.shape[1],input.shape[2]))
        input=torch.reshape(input,(input.shape[0],3,get_side_length(),get_side_length()))
        #input=(input-self.mean)/self.std
        global queries
        queries+=1
        logit=self.model(input)
        _, predict = torch.max(logit, 1)
        return predict


class my_opt:
    def __init__(self,model,samples,labels,input_device):
        global device
        device=input_device
        self.samples=samples
        self.opt=OPT_attack(my_model(model))
        self.labels=labels
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        criterion = get_criterion(self.labels)
        self.best_advs = init_attack.run(model, self.samples, criterion, early_stop=None)
    def attack(self):
        sample_adv_list=torch.zeros_like(self.samples)
        intermediates=[]
        i=0
        max_length=0
        for sample,label,init_adv in zip(self.samples,self.labels,self.best_advs):
            global queries
            queries=0
            sample=torch.reshape(sample,(3,get_side_length(),get_side_length()))
            sample=sample.cpu().numpy()
            perturbed,intermediate=self.opt.attack_untargeted(sample,label,init_adv)
            sample_adv_list[i]=perturbed
            intermediate=np.array(intermediate)
            intermediates.append(intermediate)
            i+=1
            if max_length<intermediate.shape[0]:
                max_length = intermediate.shape[0]
        intermediates=np.array(intermediates)
        return sample_adv_list,intermediates,max_length
