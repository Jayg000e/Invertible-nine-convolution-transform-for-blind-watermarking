# Thanks

The open-source repository on GitHub, https://github.com/guofei9987/blind_watermark, provides a highly extensible blind watermarking open-source repository, allowing me to focus on implementing our reversible nine-convolution transformation method without worrying about implementing irrelevant details.

#  What is the difference between the method used in this repository and the original repository?

To help readers quickly find the code corresponding to the changes made to the original method in this article, here is a detailed explanation.

The original open-source repository uses dwt in the read_img function in blind_watermark/blind_watermark.py, while this article uses generalize_dwt.

The idwt used in the embed function is changed to generalize_idwt used in this article.

The convolution kernels corresponding to these two transformations are implemented using the generate_kernels function.

The one written at the front is the frequency domain reversible nine-convolution, and the one commented out at the back is the orthogonal reversible nine-convolution.

If you want to use the latter method, just comment out the previous function and uncomment the following function.

All changes made by this article to the original method are reflected in the four functions implemented on lines 208-277 of blindwatermark/blindwatermark.py.

The pictures in the examples/output folder are experimental results of orthogonal decomposition nine-convolution, and the pictures in the examples/output_method2 folder are experimental results of frequency domain decomposition nine-convolution.

# How to reproduce the results in the report? 

1.Set up the environment required in the requirements.

2.Enter the examples folder.

3.Run python origin.py.

4.If you need to perform related attacks and view experimental results, run the relevant attack files, such as performing a pepper attack: python pepper.py.

5.The original image and watermark are in the examples/pic folder.
