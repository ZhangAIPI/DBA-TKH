import torch
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import numpy as np
import random
import sys

sys.path.append("..")
from dba_attack_utilsV6 import get_label, get_max_queries, get_threshold, get_dba_max_iter_num_in_2d, get_dim_num, \
    get_subspace_factor
import time

global device


# 对原样本添加噪声，获取一个随机的对抗样本
def get_x_adv(x_o: torch.Tensor, label: torch.Tensor, model) -> torch.Tensor:
    criterion = get_criterion(label)
    init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=100)
    x_adv = init_attack.run(model, x_o, criterion)
    return x_adv


# 根据原样本和对抗样本，获取u，不考虑数据类型的情况下x_o+u=x_adv
def get_difference(x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    difference = x_adv - x_o
    if torch.norm(difference, p=2) == 0:
        raise ('difference is zero vector!')
        return difference
    return difference


"""def get_foot(x_o: torch.Tensor, x_o2x_adv: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    alpha = torch.sum(direction * x_o2x_adv) / torch.sum(direction * direction)
    foot = x_o + alpha * direction
    if alpha==0:
        print('pos6')
    if torch.all(x_o==foot):
        print('pos9')
    return foot


def symmetric(x_o: torch.Tensor, x: torch.Tensor, x_o2x_adv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    alpha = torch.sum(x_o2x_adv * (x - x_o)) / torch.sum(x_o2x_adv * x_o2x_adv)
    x_s = 2 * (x_o + alpha * x_o2x_adv) - x
    direction = x_s - x_o
    if torch.norm(direction)==0:
        print('pos8')
    direction = direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv)
    if torch.any(torch.isnan(direction)):
        print('pos7')
    return x_s, direction"""


def rotate_in_2d(x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8) -> torch.Tensor:
    alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
    orthogonal = direction - alpha * x_o2x_adv
    direction_theta = x_o2x_adv + torch.norm(x_o2x_adv, p=2) / torch.norm(orthogonal, p=2) * orthogonal * np.tan(theta)
    direction_theta = direction_theta / torch.norm(direction_theta) * torch.norm(x_o2x_adv)
    return direction_theta


def get_orthogonal_1d(x_o2x_adv: torch.Tensor, n) -> torch.Tensor:
    random.seed(time.time())
    direction = torch.zeros(x_o2x_adv.shape, device=device)
    total_dimension = x_o2x_adv.shape[0] * x_o2x_adv.shape[1] * x_o2x_adv.shape[2] * x_o2x_adv.shape[3]
    for i in range(n):
        pos = random.randrange(0, total_dimension)
        c = pos // (x_o2x_adv.shape[2] * x_o2x_adv.shape[3])
        h = pos % (x_o2x_adv.shape[2] * x_o2x_adv.shape[3]) // x_o2x_adv.shape[3]
        w = pos % x_o2x_adv.shape[3]
        while x_o2x_adv[0, c, h, w] == 0 or is_visited_1d[pos] == 1:
            pos = random.randrange(0, total_dimension)
            c = pos // (x_o2x_adv.shape[2] * x_o2x_adv.shape[3])
            h = pos % (x_o2x_adv.shape[2] * x_o2x_adv.shape[3]) // x_o2x_adv.shape[3]
            w = pos % x_o2x_adv.shape[3]
        # is_visited_1d[pos] == 1
        direction[0, c, h, w] = 2 * torch.rand(1, device=device) - 1
    if torch.sum(direction * x_o2x_adv) < 0:
        direction = -direction
    direction = rotate_in_2d(x_o2x_adv, direction, theta=np.pi / 2)
    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2)


def get_orthogonal_1d_in_subspace(x_o2x_adv: torch.Tensor, n) -> torch.Tensor:
    random.seed(time.time())

    original_shape = list(x_o2x_adv.shape)
    subspace_shape = original_shape.copy()
    subspace_shape[2] //= get_subspace_factor()
    subspace_shape[3] //= get_subspace_factor()

    direction = torch.zeros(subspace_shape, device=device)
    total_dimension = subspace_shape[0] * subspace_shape[1] * subspace_shape[2] * subspace_shape[3]
    for i in range(n):
        pos = random.randrange(0, total_dimension)
        c = pos // (subspace_shape[2] * subspace_shape[3])
        h = pos % (subspace_shape[2] * subspace_shape[3]) // subspace_shape[3]
        w = pos % subspace_shape[3]
        while x_o2x_adv[0, c, h, w] == 0 or is_visited_1d[pos] == 1:
            pos = random.randrange(0, total_dimension)
            c = pos // (subspace_shape[2] * subspace_shape[3])
            h = pos % (subspace_shape[2] * subspace_shape[3]) // subspace_shape[3]
            w = pos % subspace_shape[3]
        # is_visited_1d[pos] == 1
        direction[0, c, h, w] = 2 * torch.rand(1, device=device) - 1

    direction=direction.resize_(original_shape)

    if torch.sum(direction * x_o2x_adv) < 0:
        direction = -direction
    direction = rotate_in_2d(x_o2x_adv, direction, theta=np.pi / 2)
    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2)


def get_v_2d(u: torch.Tensor, n) -> torch.Tensor:
    if torch.norm(u) == 0:
        print("Error! u=0!")
    total_dimension = u.shape[3] * u.shape[2]
    v = torch.zeros(u.shape, device=device)
    pos = np.arange(total_dimension)
    selected_pos = np.random.choice(pos, n, replace=False, p=probability / np.sum(probability))
    h_array = selected_pos // u.shape[3]
    w_array = selected_pos % u.shape[3]
    flag = 1
    for i in range(n):
        v[0, :, h_array[i], w_array[i]] = torch.rand(3)
        if flag and torch.any(u[0, :, h_array[i], w_array[i]] != 0):
            h, w = h_array[i], w_array[i]
            flag = 0
    for flag in range(3):
        if u[0, flag, h, w] != 0:
            break;
    v[0, flag, h, w] = (u[0, flag, h, w] * v[0, flag, h, w] - torch.sum(u * v)) / u[0, flag, h, w]
    # v = v - torch.sum(u * v) / torch.sum(u * u) * u
    return v / torch.norm(v, p=2), h_array, w_array


# 二维点是否合法
def is_position_valid(h: int, w: int) -> bool:
    return 0 <= h < selected_h and 0 <= w < selected_w


def update_probability_neighbor1(h: int, w: int, success: bool, p: np.double):
    probability[selected_w * h + w] += (2 * success - 1) * p
    probability[selected_w * h + w] = max(0, probability[selected_w * h + w])


def update_probability_neighbor2(h: int, w: int, success: bool, p: np.double):
    probability[selected_w * h + w] += (2 * success - 1) * p * p
    probability[selected_w * h + w] = max(0, probability[selected_w * h + w])


# 根据迭代中间结果更新概率矩阵
def update_probability(h_array: np.ndarray, w_array: np.ndarray, success: bool, p: np.double):
    for h, w in zip(h_array, w_array):

        h -= 1
        w -= 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)
        w -= 1
        if is_position_valid(h, w):
            update_probability_neighbor1(h, w, success, p)

        h -= 1
        w -= 2
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w += 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        h -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)
        w -= 1
        if is_position_valid(h, w):
            update_probability_neighbor2(h, w, success, p)


# 在一个维度下，获取最佳的对抗样本x_hat
def get_x_hat_in_2d(x_o: torch.Tensor, x_adv: torch.Tensor, axis_unit1: torch.Tensor, axis_unit2: torch.Tensor,
                    net: torch.nn.Module, queries, original_label, init_theta=np.pi / 16):
    d = torch.norm(x_adv - x_o, p=2)

    # 判断pi/16是否是对抗样本点，若不是直接退出这个平面
    theta = init_theta
    x_hat = x_adv
    right_theta = np.pi / 2
    x = x_o + d * (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) * np.cos(theta)
    label = get_label(net(x))
    queries += 1
    if label != original_label:
        x_hat = x
        left_theta = theta
        flag = 1
    else:
        x = x_o + d * (axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)) * np.cos(theta)
        label = get_label(net(x))
        queries += 1
        if label != original_label:
            x_hat = x
            left_theta = theta
            flag = -1
        else:
            return x_hat, queries, False

    # 二分法找到最佳的theta，进而找到最佳的x_hat
    theta = (left_theta + right_theta) / 2
    for i in range(get_dba_max_iter_num_in_2d()):
        x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) * np.cos(theta)
        label = get_label(net(x))
        queries += 1
        if label != original_label:
            left_theta = theta
            x_hat = x
            return x_hat, queries, True
        else:
            flag = -flag
            x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) * np.cos(theta)
            label = get_label(net(x))
            queries += 1
            if label != original_label:
                left_theta = theta
                x_hat = x
                return x_hat, queries, True
            else:
                right_theta = theta
        theta = (left_theta + right_theta) / 2

    return x_hat, queries, True


# 根据原样本x_o和预先训练好的判别器net，获取最佳的对抗样本x_hat
def get_x_hat(x_o: torch.Tensor, net: torch.nn.Module, original_label, init_x=None, delta_coefficient=0.):
    if get_label(net(x_o)) != original_label:
        return x_o, 0
    if init_x is None:
        x_adv = get_x_adv(x_o, original_label, net)
    else:
        x_adv = init_x
    x_hat = x_adv
    queries = 0.
    unchanged_times = 0
    init_theta = np.pi / 16
    delta = torch.zeros(x_o.shape, device=device)
    dist = torch.torch.norm(x_o - x_adv)
    intermediate = []
    intermediate.append([0, dist.item()])

    while queries < get_max_queries() and dist > get_threshold():

        x_o2x_adv = get_difference(x_o, x_adv)
        axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv)
        direction = get_orthogonal_1d_in_subspace(x_o2x_adv, get_dim_num())
        axis_unit2 = direction / torch.norm(direction)
        x_hat, queries, changed = get_x_hat_in_2d(x_o, x_adv, axis_unit1, axis_unit2, net, queries, original_label,
                                                  init_theta)

        if changed:
            unchanged_times = 0
            init_theta *= 1.05
            # update_probability(h_array, w_array, success=changed, p=0.5)
        else:
            unchanged_times += 1
            init_theta *= 0.99

        x_hat += delta * delta_coefficient
        if get_label(net(x_hat)) == original_label:
            x_hat -= delta * delta_coefficient
            unchanged_times += 1
        else:
            unchanged_times = 0
        delta = delta * delta_coefficient + x_hat - x_adv
        x_adv = x_hat
        queries += 1

        if unchanged_times >= 10:
            p = torch.ones(x_adv.shape, device=device) * 0.999
            s = torch.bernoulli(p) * 2 - 1
            difference = get_difference(x_o, x_adv)
            x_try = difference * s + x_o
            if get_label(net(x_try)) != original_label:
                x_adv = x_try
                queries += 1
                unchanged_times = 0
                # delta.zero_()
            else:
                x_try = difference * s * 1.2 + x_o
                if get_label(net(x_try)) != original_label:
                    x_adv = x_try
                    unchanged_times = 0
                    # delta.zero_()
                queries += 2
        x_hat = x_adv
        dist = torch.norm(x_hat - x_o)
        intermediate.append([queries, dist.item()])
        if queries >= get_max_queries() or dist <= get_threshold():
            print(dist)
            break
    return x_hat, queries, intermediate


class BDAattack:
    def __init__(self, model, input_device):
        self.net = model
        global device
        device = input_device

    def attack(self, inputs, labels, delta_coefficient=0.):
        x_adv_list = torch.zeros_like(inputs)
        queries = []
        intermediates = []
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        criterion = get_criterion(labels)
        best_advs = init_attack.run(self.net, inputs, criterion, early_stop=None)
        max_length = 0
        for i, [input, label] in enumerate(zip(inputs, labels)):
            global probability
            probability = np.ones(input.shape[1] * input.shape[2])
            global is_visited_1d
            is_visited_1d = torch.zeros(input.shape[0] * input.shape[1] * input.shape[2])
            global selected_h
            global selected_w
            selected_h = input.shape[1]
            selected_w = input.shape[2]
            x_adv, q, intermediate = get_x_hat(input[np.newaxis, :, :, :], self.net, label.reshape(1, ).to(device),
                                               init_x=best_advs[i][np.newaxis, :, :, :],
                                               delta_coefficient=delta_coefficient)
            x_adv_list[i] = x_adv[0]
            queries.append(q)
            intermediate = np.array(intermediate)
            intermediates.append(intermediate)
            if max_length < intermediate.shape[0]:
                max_length = intermediate.shape[0]
        intermediates = np.array(intermediates)
        queries = np.array(queries)
        return x_adv_list, queries, intermediates, max_length
