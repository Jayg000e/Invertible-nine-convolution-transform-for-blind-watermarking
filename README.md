# 关于代码

github开源仓库https://github.com/guofei9987/blind_watermark
提供了一个扩展性高的盲水印开源仓库，使得我可以专注于实现我们的可逆九卷积变换方法，而不费心去实现一些无关的细节和实现一些繁琐的代码。

# 关于创新点

为方便读者在快速找到本文的创新点对应的代码，这里作出详细说明。

原开源仓库在blind_watermark/blind_watermark.py中read_img函数中使用的是dwt，本文使用的是generalize_dwt，

embed函数中使用的idwt改为本文中使用的generalize_idwt。

这两个变换对应的卷积核生成利用generate_kernels函数予以实现，

写在前面的是频域可逆九卷积，写在后面被注释掉的是正交可逆九卷积，

如果要使用后面一种方法只需注释掉前面的函数，并取消后面函数的注释。

本文的所有创新点体现在blindwatermark/blindwatermark.py中的第208行-第277行实现的四个函数。 

examples/output 文件夹中的图片是正交分解九卷积的实验结果，examples/output_method2文件夹中的图片是频域分解九卷积的实验结果。

# 关于如何复现报告中的结果

1.配置requirements中需要的环境

2.进入examples文件夹

3.python origin.py

4.如果需要进行相关攻击，并查看实验结果，运行相关攻击文件，如进行椒盐攻击：
   python pepper.py

5.原始图片和水印在examples/pic文件夹中
