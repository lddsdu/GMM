# -*- coding: utf-8 -*-
# @Time    : 2020-05-04 12:56
# @Author  : lddsdu
# @File    : img2gif.py

import os
import cv2
import imageio

out_filename = "gd.gif"  # 转化的GIF图片名称
filenames = ["sgd-{:0>2}.jpg".format(i) for i in ([0] + [j * 30 + 9 for j in range(30)])]

frames = []
for image_name in filenames:
    im = cv2.imread(os.path.join("data", image_name))[:, :, ::-1]
    frames.append(im)

imageio.mimsave(out_filename, frames, duration=0.3)
