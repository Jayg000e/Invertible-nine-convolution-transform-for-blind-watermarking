# -*- coding: utf-8 -*-
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att
import numpy as np
import cv2

# %%椒盐攻击
att.salt_pepper_att('output_method2/embedded.png', 'output_method2/pepper.png', ratio=0.05)
# ratio是椒盐概率

# %%纵向裁剪打击.png
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename='output_method2/pepper.png', wm_shape=(45, 68), out_wm_name='output_method2/pepper_extracted.png')

