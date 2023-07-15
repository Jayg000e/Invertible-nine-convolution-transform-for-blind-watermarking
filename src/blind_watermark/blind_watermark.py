#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2020/8/13
# @Author  : github.com/guofei9987

import numpy as np
import copy
import cv2
from pywt import dwt2, idwt2
from .pool import AutoPool


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(4, 4), mode='common', processes=None):
        self.block_shape = np.array(block_shape)
        self.password_wm, self.password_img = password_wm, password_img  # 打乱水印和打乱原图分块的随机种子
        self.d1, self.d2 = 25,10  # d1/d2 越大鲁棒性越强,但输出图片的失真越大

        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dwt 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = AutoPool(mode=mode, processes=processes)
        self.kernels,self.inv_kernels=self.generate_kernels()


    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img(self, filename):
        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        # self.img = cv2.resize(cv2.imread(filename),(2520,1440)).astype(np.float32)
        self.img = cv2.imread(filename).astype(np.float32)
        bordersize=[]
        for i in range(2):
            if self.img.shape[i]% 3==0:
                bordersize.append(0)
            else:
                bordersize.append(3-self.img.shape[i]% 3)
        self.img = cv2.resize(self.img,(self.img.shape[1]+bordersize[1],self.img.shape[0]+bordersize[0])).astype(np.float32)
        self.img_shape = self.img.shape[:2]
        # 如果不是偶数，那么补上白边
        # self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
        #                                   0, 3-self.img.shape[0] % 3, 0, 3-self.img.shape[1] % 3,
        #                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.img_YUV = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)

        self.ca_shape = [(i + 1) // 3 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            feature=self.generalize_dwt(self.img_YUV[:, :, channel])
            self.ca[channel], self.hvd[channel] = feature[0],feature[1:]
            # self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_img_wm(self, filename):
        # 读入图片格式的水印，并转为一维 bit 格式
        self.wm = cv2.imread(filename)[:, :, 0]
        # 加密信息只用bit类，抛弃灰度级别
        self.wm_bit = self.wm.flatten() > 128

    def read_wm(self, wm_content, mode='img'):
        if mode == 'img':
            self.read_img_wm(filename=wm_content)
        elif mode == 'str':
            byte = bin(int(wm_content.encode('utf-8').hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        else:
            self.wm_bit = np.array(wm_content)
        self.wm_size = self.wm_bit.size
        # 水印加密:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

    def block_add_wm(self, arg):
        block, shuffler, i = arg
        # dct->flatten->加密->逆flatten->svd->打水印->逆svd->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = cv2.dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        # block_dct[3,3] = (block_dct[3,3] // self.d0 + 1 / 4 + 1 / 2 * wm_1) * self.d0

        U, s, V = np.linalg.svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2
        block_dct_flatten = np.dot(U, np.dot(np.diag(s), V)).flatten()

        #block_dct_flatten = block_dct_shuffled.flatten()

        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return cv2.idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self, filename):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3
        self.idx_shuffle = np.random.RandomState(self.password_img) \
            .random(size=(self.block_num, self.block_shape[0] * self.block_shape[1])) \
            .argsort(axis=1)

        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]
            # 4维分块变回2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = self.generalize_idwt([embed_ca[channel]]+self.hvd[channel])
            # embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]].astype('float32')
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        cv2.imwrite(filename, embed_img)
        return embed_img

    def block_get_wm(self, args):
        block, shuffler = args
        # dct->flatten->加密->逆flatten->svd->解水印
        block_dct=cv2.dct(block)

        block_dct_shuffled =  block_dct.flatten()[shuffler].reshape(self.block_shape)

        # wm_dct = (block_dct[3,3] % self.d0 > self.d0 / 2) * 1
        U, s, V = np.linalg.svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4

        # wm=(wm_dct+wm)/2

        return wm

    def extract_raw(self, filename):
        # 每个分块提取 1 bit 信息
        self.read_img(filename)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来
        self.idx_shuffle = np.random.RandomState(self.password_img) \
            .random(size=(self.block_num, self.block_shape[0] * self.block_shape[1])) \
            .argsort(axis=1)
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename, wm_shape, out_wm_name=None, mode='img'):
        self.wm_size = np.array(wm_shape).prod()

        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_raw(filename=filename)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        # 解密：
        wm = self.extract_decrypt(wm_avg=wm_avg)
        # 转化为指定格式：
        if mode == 'img':
            cv2.imwrite(out_wm_name, 255 * wm.reshape(wm_shape[0], wm_shape[1]))
        elif mode == 'str':
            byte = ''.join((np.round(wm)).astype(np.int).astype(np.str))
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8')

        self.extracted=wm

        return wm

    def generalize_dwt(self,img):
        h=img.shape[0]
        w=img.shape[1]
        feature=list(np.zeros((9,h//3,w//3)))
        for i in range(h//3):
            for j in range(w//3):
                block=img[i*3:(i+1)*3,j*3:(j+1)*3]
                for k,kernel in enumerate(self.kernels):
                    feature[k][i][j]=(block*kernel).sum()
        return feature

    def generalize_idwt(self,feature):
        h=feature[0].shape[0]
        w=feature[0].shape[1]
        img=np.zeros((3*h,3*w))
        for i in range(h):
            for j in range(w):
                feature_block=np.array([[feature[0][i][j],feature[1][i][j],feature[2][i][j]],
                                        [feature[3][i][j],feature[4][i][j],feature[5][i][j]],
                                        [feature[6][i][j],feature[7][i][j],feature[8][i][j]]])
                img[3*i:3*(i+1),3*j:3*(j+1)]=np.array([(feature_block*inv_kernel).sum() for inv_kernel in self.inv_kernels]).reshape(3,3)
        return img

    #频域可逆九卷积核
    def generate_kernels(self):

        from scipy.linalg import null_space
        mean=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
        gauss=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
        prewitt_h=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        prewitt_v=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        sobel_h=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        sobel_v=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        laplace_1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        #laplace_2=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

        filter_mat=np.concatenate((mean.reshape(1,9),
                                   gauss.reshape(1,9),
                                   prewitt_h.reshape(1,9),
                                   prewitt_v.reshape(1,9),
                                   sobel_h.reshape(1,9),
                                   sobel_v.reshape(1,9),
                                   laplace_1.reshape(1,9),
                                   ),axis=0)

        pseudo_filter=null_space(filter_mat).reshape(2,9)
        filters=np.concatenate((filter_mat,pseudo_filter),axis=0)
        kernels=[filter.reshape([3,3]) for filter in list(filters)]
        inv_kernels=[inv_filter.reshape([3,3]) for inv_filter in list(np.linalg.inv(filters))]

        return kernels,inv_kernels


    #正交分解九卷积核，如果要使用这个核，把注释去掉，并把上面的方法注释掉
    # def generate_kernels(self):
    #     from sympy import GramSchmidt,Matrix
    #
    #     original=np.eye(9)
    #
    #     original[0,]=np.ones((1,9))
    #
    #     Mat_ori=[Matrix(ori) for ori in original]
    #
    #     filters=np.array(GramSchmidt(Mat_ori)).reshape(9,9).astype('float64')
    #     filters=np.array([filter/np.linalg.norm(filter) for filter in list(filters)])
    #
    #     kernels=[filter.reshape([3,3]) for filter in list(filters)]
    #     inv_kernels=[inv_filter.reshape([3,3]) for inv_filter in list(np.linalg.inv(filters))]
    #
    #     return kernels,inv_kernels
