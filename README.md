# UnDIP
UnDIP: Hyperspectral Unmixing Using Deep Image Prior
======================================================
UnDIP is a deep learning-based technique for the linear hyperspectral unmixing problem. The proposed method contains two main steps. First, the endmembers are extracted using a geometric endmember extraction method, i.e., a  simplex volume maximization in the subspace of the dataset. Then, the abundances are estimated using a deep image prior. The proposed deep image prior uses a convolutional neural network to estimate the fractional abundances, relying on the extracted endmembers and the observed hyperspectral dataset. The proposed method is evaluated on simulated and three real remote sensing data for a range of SNR values (i.e., from 20 to 50 dB). The results show considerable improvements compared to state-of-the-art methods.

If you use this code please cite the following paper
Rasti, B.,  Koirala, B., Scheunders, P., and Ghamisi, P., 
"UnDIP: Hyperspectral Unmixing Using Deep Image Prior" 
IEEE Transactions on Geoscience and Remote Sensing

Not that the copyright of the DIP software (https://github.com/DmitryUlyanov/deep-image-prior) used in UnDIP which is uploaded here is preseved.

Here, the ground truth abundances (right) are compared with the estimated ones using UnDIP (left) for Samson and Jasper Ridge Datasets.

![SamsonGIF](https://user-images.githubusercontent.com/61419984/109668183-34114500-7b71-11eb-926c-f27c833170c9.gif) 

![JasperGIF](https://user-images.githubusercontent.com/61419984/109668460-80f51b80-7b71-11eb-86a8-ab0976486c53.gif)



