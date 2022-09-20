from .utils.denoising_utils import *
import numpy as np
from skimage._shared import *
from skimage.util import *
from skimage.metrics.simple_metrics import _as_floats
from skimage.metrics.simple_metrics import mean_squared_error
from skimage._shared.utils import * 
from pymf.dist import l1_distance
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
import torch
def compare_snr(image_true, image_test, *, data_range=None):
    """
    Compute the signal to noise ratio (PSNR) for an image.
    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.", stacklevel=2)
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    
    return 10 * np.log10((np.mean(image_test ** 2, dtype=np.float64)) / err)
 
def find_endmember(EE,E):
    """
    Find the closest matches to E from EE in terms of l_2 norm
    """
    n1=EE.shape[1];
    n2=E.shape[1];
    error=np.zeros((1,n1))
    index=np.zeros((n2))
    for i in range(n2):
        for j in range(n1):
            error[0,j]= np.linalg.norm(E[:,i]-EE[:,j],2)
        b=np.argmin(error,axis=1)
        index[i] = b
    E_est = EE[:,index[0:n2].astype(int)]
    return E_est
def add_noise(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    """   
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
    """
    img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)

    return img_noisy_np
def Eucli_dist(x,y):
    a=np.subtract(x, y)
    return np.dot(a.T,a)
def Endmember_reorder(x):
    [D,N]=x.shape
    a=np.zeros((N,1))
    for i in range(N):
        a[i,0]=np.dot(x[:,i].T,x[:,i])
    per=np.argsort(a,axis=0)
   # I=np.sort(a)
    return per
def Endmember_reorder2(A,E1):
    index = []
    _,p=A.shape
    error=np.zeros((1,p))
    for l in range(p):
        for n in range(p):
            error[0,n] = Eucli_dist(A[:,l],E1[:,n])
        b=np.argmin(error)
        index=np.append(index,b) 
    index=index.astype(int)
    return index 
    
# def Vec_1d(x):
#     [D,N]=x.shape
#     if D==1:
#         y=np.reshape(x,(1,N))
#     elif N==1:
#         y=np.reshape(x,(D,1))
#     return y
def Endmember_extract(x,p):

    [D,N]=x.shape
# If no distf given, use Euclidean distance function
    Z1=np.zeros((1,1))
    O1=np.ones((1,1))
# Find farthest point
    d=np.zeros((p,N))
    I=np.zeros((p,1))
    V=np.zeros((1,N))
    ZD=np.zeros((D,1))
# if nargin<4
    for i in range(N):
        d[0,i]=Eucli_dist(x[:,i].reshape(D,1),ZD)
       # d[0,i]=l1_distance(x[:,i].reshape(D,1),ZD)
# else
#     for i=1:N
#         d(1,i)=distf(x(:,i),zeros(D,1),opt);

    I=np.argmax(d[0,:])

#if nargin<4
    for i in range(N):
       d[0,i] = Eucli_dist(x[:,i].reshape(D,1),x[:,I].reshape(D,1))
       # d[0,i] = l1_distance(x[:,i].reshape(D,1),x[:,I].reshape(D,1))

# else
#     for i=1:N
#         d(1,i)=distf(x(:,i),x(:,I(1)),opt);
    for v in range(1,p):
        #D=[d[0:v-2,I] ; np.ones((1,v-1)) 0]
        D1=np.concatenate((d[0:v,I].reshape((v,I.size)), np.ones((v,1))),axis=1)
        D2=np.concatenate((np.ones((1,v)),Z1),axis=1)
        D4=np.concatenate((D1,D2),axis=0)
        D4=np.linalg.inv(D4)
        for i in range(N):
            D3=np.concatenate((d[0:v,i].reshape((v,1)), O1),axis=0)
            V[0,i]=np.dot(np.dot(D3.T,D4),D3)
        
        I=np.append(I,np.argmax(V))
        # if nargin<4
        for i in range(N):
            #d[v,i]=l1_distance(x[:,i].reshape(D,1),x[:,I[v]].reshape(D,1))
            d[v,i]=Eucli_dist(x[:,i].reshape(D,1),x[:,I[v]].reshape(D,1))
            
        # else
        #     for i=1:N
        #         d(v,i)=distf(x(:,i),x(:,I(v)),opt);
    per=np.argsort(I)
    I=np.sort(I)
    d=d[per,:]
    return I, d

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
def torch_dim(x,dim):
    return torch.sqrt(torch.sum(x*x,dim));
