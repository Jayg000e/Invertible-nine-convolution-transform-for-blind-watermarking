# -*- coding: utf-8 -*-
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att
import numpy as np

# 一次横向裁剪打击
att.cut_att_width('output_method2/embedded.png', 'output_method2/cutattack.png', ratio=0.5)
att.anti_cut_att('output_method2/cutattack.png', 'output_method2/cutattack_fill.png', origin_shape=(720, 1281))

# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename="output_method2/cutattack_fill.png", wm_shape=(45, 68), out_wm_name="output_method2/cutattack_extracted.png")
