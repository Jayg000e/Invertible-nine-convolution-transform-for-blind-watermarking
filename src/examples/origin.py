#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")



from blind_watermark import WaterMark
bwm1 = WaterMark(password_wm=1, password_img=1)


# 读取原图
bwm1.read_img('pic/beautiful_scene.jpg')
# 读取水印
bwm1.read_wm('pic/guojia.png')
# 打上盲水印
bwm1.embed('output_method2/embedded.png')


# %% 解水印


bwm1 = WaterMark(password_wm=1, password_img=1)
# 注意需要设定水印的长宽wm_shape
bwm1.extract(filename='output_method2/embedded.png', wm_shape=(45, 68), out_wm_name='output_method2/wm_extracted.png', )


