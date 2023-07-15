# -*- coding: utf-8 -*-
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att
# 缩放攻击
att.resize_att('output_method2/embedded.png', 'output_method2/resizeattack.png', out_shape=(800, 600))
att.resize_att('output_method2/resizeattack.png', 'output_method2/resizeattack_reset.png', out_shape=(720, 1281))
# out_shape 是分辨率，需要颠倒一下
# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename="output_method2/resizeattack_reset.png", wm_shape=(45, 68), out_wm_name="output_method2/resizeattack_extracted.png")
