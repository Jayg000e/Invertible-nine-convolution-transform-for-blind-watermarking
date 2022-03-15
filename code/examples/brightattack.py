# -*- coding: utf-8 -*-
# run origin.py to generate the embedded image
import sys
sys.path.append("..")
from blind_watermark import att

# 亮度调高攻击
att.bright_att('output_method2/embedded.png', 'output_method2/brightattack.png', ratio=1.1)

#%% 提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename='output_method2/brightattack.png', wm_shape=(45, 68), out_wm_name='output_method2/brightattack_extracted.png')
