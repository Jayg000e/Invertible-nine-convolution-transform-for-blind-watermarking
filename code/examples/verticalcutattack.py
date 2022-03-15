# -*- coding: utf-8 -*-
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att
import numpy as np

# 一次纵向裁剪打击
att.cut_att_height('output_method2/embedded.png', 'output_method2/verticalcutattack.png', ratio=0.5)

att.anti_cut_att('output_method2/verticalcutattack.png', 'output_method2/verticalcutattack_fill.png', origin_shape=(720, 1281))

# %%纵向裁剪打击.png
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename="output_method2/verticalcutattack_fill.png", wm_shape=(45, 68), out_wm_name="output_method2/verticalcutattack_extracted.png")
