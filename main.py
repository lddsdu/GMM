# -*- coding: utf-8 -*-
# @Time    : 2020-05-03 13:55
# @Author  : lddsdu
# @File    : main.py

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
import torch.nn as nn


def random_normalize(num, dim=2):
    x = np.random.randn(num, dim)
    return x


def transform(x, mu, sigma=None):
    if sigma is not None:
        x = np.dot(x, sigma)

    x = x + mu
    return x


def draw_ellipse(ax, sigma1, sigma2, xy, facecolor="yellow"):
    ell1 = Ellipse(xy=xy, width=sigma1, height=sigma2, angle=0,
                   facecolor=facecolor, alpha=0.5)
    ax.add_patch(ell1)


# all parameters
origin_x = random_normalize(100)
mu_a = np.asarray([1.0, 1.0])
sigma_a = np.asarray([[6.0, 0.0], [0.0, 1.0]])
clutter_a = transform(origin_x, mu_a, sigma_a)

origin_x = random_normalize(100)
mu_b = np.asarray([-5.0, -5.0])
sigma_b = np.asarray([[1.0, 0.0], [0.0, 1.0]])
clutter_b = transform(origin_x, mu_b, sigma_b)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(clutter_a[:, 0], clutter_a[:, 1], marker="o", c="g")
plt.scatter(clutter_b[:, 0], clutter_b[:, 1], marker="o", c="r")
plt.title("Origin Distribution")
# draw_ellipse(ax, 4, 2, xy=(1, 1), facecolor="green")
# draw_ellipse(ax, 3, 3, xy=(-5, -5), facecolor="red")
plt.savefig("data/origin.jpg")


def gauss(x1, x2, u1, u2, s1, s2):
    v1 = 1 / (2 * math.pi * s1 * s2)
    v2 = math.exp(- 0.5 * (((x1 - u1) / s1) ** 2 + ((x2 - u2) / s2) ** 2))
    return v1 * v2


def gauss_torch(x, mu_k, sigma_k):
    v1 = torch.tensor([1.0]) / (2 * math.pi * sigma_k[0] * sigma_k[1])  # (1)
    v2 = torch.exp(-0.5 * (torch.sum(((x - mu_k) / sigma_k) ** 2, dim=1))) + 1e-20  # (n, 1)
    return v1.unsqueeze(0) * v2


def em(ca, cb, step=100, interval=10):
    cab = np.concatenate([ca, cb], axis=0)  # N_a + N_b, 2

    # 初始化的参数
    hyp_mu_a = np.asarray([-6.0, 2.0])
    hyp_mu_b = np.asarray([6.0, -6.0])
    hyp_sigma_a = np.asarray([5.0, 0.5])
    hyp_sigma_b = np.asarray([2.0, 0.5])
    w1 = 0.5

    fig = plt.figure()
    plt.scatter(clutter_a[:, 0], clutter_a[:, 1], marker="o", c="g")
    plt.scatter(clutter_b[:, 0], clutter_b[:, 1], marker="o", c="r")
    plt.title("Step {}".format(0))

    ax = fig.add_subplot(111)
    draw_ellipse(ax, hyp_sigma_a[0], hyp_sigma_a[1],
                 xy=(hyp_mu_a[0], hyp_mu_a[1]), facecolor="green")
    draw_ellipse(ax, hyp_sigma_b[0], hyp_sigma_b[1],
                 xy=(hyp_mu_b[0], hyp_mu_b[1]), facecolor="red")
    plt.savefig("data/s-{:0>2}.jpg".format(0))

    for s in range(step):
        print("step: {}".format(s + 1))
        # E-step: 求解隐变量的后验分布
        # q-matrix, 存储先验分布 shape: (N, K)
        gamma = np.zeros((ca.shape[0] + cb.shape[0], 2))

        for n in range(ca.shape[0]):
            gamma[n][0] = w1 * gauss(ca[n][0], ca[n][1], hyp_mu_a[0], hyp_mu_a[1], hyp_sigma_a[0] ** 0.5,
                                     hyp_sigma_a[1] ** 0.5)
            gamma[n][1] = (1 - w1) * gauss(ca[n][0], ca[n][1], hyp_mu_b[0], hyp_mu_b[1], hyp_sigma_b[0] ** 0.5,
                                           hyp_sigma_b[1] ** 0.5)

        for n in range(cb.shape[0]):
            gamma[n + ca.shape[0]][0] = w1 * gauss(cb[n][0], cb[n][1], hyp_mu_a[0], hyp_mu_a[1], hyp_sigma_a[0] ** 0.5,
                                                   hyp_sigma_a[1] ** 0.5)
            gamma[n + ca.shape[0]][1] = (1 - w1) * gauss(cb[n][0], cb[n][1], hyp_mu_b[0], hyp_mu_b[1],
                                                         hyp_sigma_b[0] ** 0.5, hyp_sigma_b[1] ** 0.5)

        # normalization
        gamma = gamma / np.expand_dims(np.sum(gamma, 1), 1)

        # M-step: maximum the ELBO, update the weight
        N = np.sum(gamma, axis=0, keepdims=True)  # (1, 2)

        hyp_mu_a = 1 / N[0][0] * (np.sum(gamma[:, 0:1] * cab, axis=0))
        hyp_mu_b = 1 / N[0][1] * (np.sum(gamma[:, 1:2] * cab, axis=0))

        hyp_sigma_a = 1 / N[0][0] * np.sum(gamma[:, 0:1] * ((cab - np.expand_dims(hyp_mu_a, axis=0)) ** 2), axis=0)
        hyp_sigma_b = 1 / N[0][1] * np.sum(gamma[:, 1:2] * ((cab - np.expand_dims(hyp_mu_b, axis=0)) ** 2), axis=0)

        w1 = N[0][0] / (ca.shape[0] + cb.shape[0])

        if s % interval == 0 or (s + 1) == step:
            fig = plt.figure()
            plt.scatter(clutter_a[:, 0], clutter_a[:, 1], marker="o", c="g")
            plt.scatter(clutter_b[:, 0], clutter_b[:, 1], marker="o", c="r")
            plt.title("Step {}".format(s + 1))

            ax = fig.add_subplot(111)
            draw_ellipse(ax, hyp_sigma_a[0], hyp_sigma_a[1],
                         xy=(hyp_mu_a[0], hyp_mu_a[1]), facecolor="blue")
            draw_ellipse(ax, hyp_sigma_b[0], hyp_sigma_b[1],
                         xy=(hyp_mu_b[0], hyp_mu_b[1]), facecolor="yellow")
            plt.savefig("data/s-{:0>2}.jpg".format(s + 1))

    return hyp_mu_a, hyp_mu_b, hyp_sigma_a, hyp_sigma_b, w1


# print(em(clutter_a, clutter_b, step=5, interval=1))


class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()
        # 初始化的参数
        self.hyp_mu_a = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)
        self.hyp_mu_b = nn.Parameter(torch.tensor([-2.0, 2.0]), requires_grad=True)
        self.hyp_sigma_a = nn.Parameter(torch.tensor([5.0, 0.5]), requires_grad=True)
        self.hyp_sigma_b = nn.Parameter(torch.tensor([2.0, 0.5]), requires_grad=True)
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

    def forward(self, ca, cb, cab):
        # previous turn parameters
        hyp_mu_a, hyp_mu_b = self.hyp_mu_a.detach().numpy(), self.hyp_mu_b.detach().numpy()
        hyp_sigma_a, hyp_sigma_b = self.hyp_sigma_a.detach().numpy(), self.hyp_sigma_b.detach().numpy()
        w = torch.softmax(self.w, dim=0).detach().numpy()
        w1, w2 = w[0], w[1]

        gamma = np.zeros((ca.shape[0] + cb.shape[0], 2), dtype=np.float)

        # E-step
        for n in range(ca.shape[0]):
            gamma[n][0] = w1 * gauss(ca[n][0], ca[n][1], hyp_mu_a[0], hyp_mu_a[1], hyp_sigma_a[0] ** 0.5,
                                     hyp_sigma_a[1] ** 0.5)
            gamma[n][1] = w2 * gauss(ca[n][0], ca[n][1], hyp_mu_b[0], hyp_mu_b[1], hyp_sigma_b[0] ** 0.5,
                                     hyp_sigma_b[1] ** 0.5)

        for n in range(cb.shape[0]):
            gamma[n + ca.shape[0]][0] = w1 * gauss(cb[n][0], cb[n][1], hyp_mu_a[0], hyp_mu_a[1], hyp_sigma_a[0] ** 0.5,
                                                   hyp_sigma_a[1] ** 0.5)
            gamma[n + ca.shape[0]][1] = w2 * gauss(cb[n][0], cb[n][1], hyp_mu_b[0], hyp_mu_b[1], hyp_sigma_b[0] ** 0.5,
                                                   hyp_sigma_b[1] ** 0.5)

        # normalization
        gamma = gamma / np.expand_dims(np.sum(gamma, 1), 1)
        gamma = torch.tensor(gamma, dtype=torch.float)

        ws = torch.softmax(self.w, dim=-1)
        # M-step TODO, 训练后期出现了inf, loss变为nan
        loss_a = (gamma[:, 0].unsqueeze(0) *
                  torch.log(ws[0] * gauss_torch(cab, self.hyp_mu_a, self.hyp_sigma_a ** 0.5))).mean()
        loss_b = (gamma[:, 1].unsqueeze(0) *
                  torch.log(ws[1] * gauss_torch(cab, self.hyp_mu_b, self.hyp_sigma_b ** 0.5))).mean()
        loss = loss_a + loss_b
        return - loss


def gd(ca, cb, cab):
    gmm = GMM()
    fig = plt.figure()
    plt.scatter(ca[:, 0], ca[:, 1], marker="o", c="g")
    plt.scatter(cb[:, 0], cb[:, 1], marker="o", c="r")
    plt.title("Step {}".format(0))

    ax = fig.add_subplot(111)
    draw_ellipse(ax, gmm.hyp_sigma_a[0].item(), gmm.hyp_sigma_a[1].item(),
                 xy=(gmm.hyp_mu_a[0].item(), gmm.hyp_mu_a[1].item()), facecolor="green")
    draw_ellipse(ax, gmm.hyp_sigma_b[0].item(), gmm.hyp_sigma_b[1].item(),
                 xy=(gmm.hyp_mu_b[0].item(), gmm.hyp_mu_b[1].item()), facecolor="red")
    plt.savefig("data/sgd-{:0>2}.jpg".format(0))

    optimizer = torch.optim.Adam(lr=0.1, params=gmm.parameters())

    for i in range(10000):
        loss = gmm.forward(ca, cb, cab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Step {}, loss {:.4f}".format(1 + i, loss.item()))

        fig = plt.figure()
        plt.scatter(ca[:, 0], ca[:, 1], marker="o", c="g")
        plt.scatter(cb[:, 0], cb[:, 1], marker="o", c="r")
        plt.title("Step {}".format(i + 1))

        ax = fig.add_subplot(111)
        draw_ellipse(ax, gmm.hyp_sigma_a[0].item(), gmm.hyp_sigma_a[1].item() ** 0.5,
                     xy=(gmm.hyp_mu_a[0].item(), gmm.hyp_mu_a[1].item()), facecolor="blue")
        draw_ellipse(ax, gmm.hyp_sigma_b[0].item(), gmm.hyp_sigma_b[1].item() ** 0.5,
                     xy=(gmm.hyp_mu_b[0].item(), gmm.hyp_mu_b[1].item()), facecolor="yellow")

        if (i + 1) % 10 == 0:
            plt.savefig("data/sgd-{:0>2}.jpg".format(i))


clutter_ab = torch.cat([torch.tensor(clutter_a, dtype=torch.float), torch.tensor(clutter_b, dtype=torch.float)], dim=0)
gd(clutter_a, clutter_b, clutter_ab)
