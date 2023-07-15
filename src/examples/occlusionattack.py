# coding=utf-8
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att
import numpy as np
import cv2

# %%
# 攻击
att.shelter_att('output_method2/embedded.png', 'output_method2/occlusionattack.png', ratio=0.1, n=10)

# %%多遮挡攻击.png
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename='output_method2/occlusionattack.png', wm_shape=(45, 68), out_wm_name='output_method2/occlusionattack_extracted.png')

