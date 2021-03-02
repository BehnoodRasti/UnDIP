# UnDIP
UnDIP: Hyperspectral Unmixing Using Deep Image Prior
======================================================
UnDIP is a deep learning-based technique for the linear hyperspectral unmixing problem. The proposed method contains two main steps. First, the endmembers are extracted using a geometric endmember extraction method, i.e., a  simplex volume maximization in the subspace of the dataset. Then, the abundances are estimated using a deep image prior. The proposed deep image prior uses a convolutional neural network to estimate the fractional abundances, relying on the extracted endmembers and the observed hyperspectral dataset. The proposed method is evaluated on simulated and three real remote sensing data for a range of SNR values (i.e., from 20 to 50 dB). The results show considerable improvements compared to state-of-the-art methods.

If you use this code please cite the following paper
Rasti, B.,  Koirala, B., Scheunders, P., and Ghamisi, P., 
"UnDIP: Hyperspectral Unmixing Using Deep Image Prior" 
IEEE Transactions on Geoscience and Remote Sensing

https://user-images.githubusercontent.com/61419984/109656246-8304ad80-7b64-11eb-8cd0-5b9b5d89c728.mp4
