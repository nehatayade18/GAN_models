# -*- coding: utf-8 -*-
"""Image_Super_Resolution_using_Autoencoders.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v9LNuFERQPbapxb9scB99KcTEdIc3Chf

# Image Super Resolution Using Auto-encoders Tutorial
---

This project shows how to train an Auto-encoder to convert low-quality images into high-quality using TensorFlow and Keras. 

## Introduction 

### Auto-encoder
An Auto-encoder is a type of Neural Network that tries to learn a representation of its input data, but in a space with much smaller dimensionality. This smaller representation is able to learn important features of the input data that can be used to later reconstruct the data. 
An auto encoder is principally composed of 3 elements: an **encoder**, a **decoder** and a **loss function**.
* Both the encoder and decoder are usually Convolutional Neural Networks.
* The encoder tries to reduce the dimensionality of the input while the decoder tries to recover our image from this new space. 
    * First, the **encoder** takes an input and passes it through its layers, gradually reducing the receptive field of the input. At the end of the encoder, the input is reduced to a *linear feature representation*.  
    * This linear feature representation is then fed to the **decoder** which tries to recover the image through *upsampling* it (increasing its receptive field) gradually until it reaches the end where the output has the same dimensions as the original input. 
* This architecture is ideal for preserving the dimensionality. However, the linear compression of the input is a *lossy* process, meaning it losses information in the process.
* The **loss function** is a way of describing a meaningful difference (or distance) between the input and output. During training, our goal is to minimize such difference so that the network will eventually lean to reconstruct the best possible output.

In order to teach an auto-encoder how to reconstruct an image, we need to show it pairs of low quality and high quality images. This way, then network will try to find the patterns and important encoded visual features needed to be able to reconstruct it from the low quality version.  

During training, the hidden layers will capture a **dense** (compressed) representation of the input data.

### Super-Resolution

Auto-encoders have many applications in image processing, especially in the Image Transformation task. Some of these applications include: 
- Denoising
- Super-Resolution
- Colorization

In this tuturial, we'll go over the problem of **super-resolution**, where the task is to generate a high-resolution output image from a low-resolution input.

## The encoder

Here we'll be designing the following encoder using TensorFlow and Keras.

<img src="https://github.com/ilopezfr/image-superres/blob/master/img/model_encoder.png?raw=1"  width="450" />

Run the complete code below, it's the entire encoder. We can see a summary of what our encoder looks like right after.
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from keras.models import Model
from keras import regularizers

# Encoder

n = 256
chan = 3
input_img = Input(shape=(n, n, chan))

l1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = MaxPooling2D(padding='same')(l5)
l3 = Dropout(0.5)(l3)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)
encoder = Model(input_img, l7)

encoder.summary()

"""Some comments: 

**Layers `l4 and l5`**: 
```python
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
```
- In Layer 3, in our new reduced space of 128x128, let's put more convolutions (128 of 3x3) so that we can learn more features. Because the space is smaller, we have less information. So by having more different convolutions we can try to compensate for the loss of information. 
- This gives the network much more perspective. We can imagine these convolution filters as being like a different point of view, each at the same thing (the image). An analogy is having many people (convolution filters) on a team working together and seeing same problem from different angles.

## The decoder

Let's now build our decoder!

The steps are pretty much the same as in the encoder but in **reverse order**.

<img src="https://github.com/ilopezfr/image-superres/blob/master/img/model_decoder.png?raw=1" width="450"/>
"""

# Decoder

l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)

l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l3 = Dropout(0.3)(l3)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)

l15 = add([l14, l2])

# chan = 3, for RGB
decoded = Conv2D(chan, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)

# Create our network
autoencoder = Model(input_img, decoded)
# You'll understand later what this is
autoencoder_hfenn = Model(input_img, decoded)

"""Some comments:

**Layers `l11` and  `l15`** :
```python
l11 = add([l5, l10])
l15 = add([l14, l2])
```
- Using a **merge layer** we are performing an "add" operation with the layer below and a layer from the encoder: 
  - `conv2D_l4 + conv2d_l7`
  - `conv2D_2 + conv2D_9`
- We do this for various reasons:
  * we want to share knowledge from the encoder to the decoder like "Hey, I'm trying to reconstruct this part of the image, did it roughly look like that to you also?".
  * it helps with the **"vanishing gradient problem"**, in which the network looses information from going deeper. Neurons at the beginning have difficulty learning in deep networks because they are too far from the deeper networks and can't have their knowledge shared.

## Complete network
*Here*'s the final network :

<img src="https://github.com/ilopezfr/image-superres/blob/master/img/model.png?raw=1"  width="450" />
"""

autoencoder.summary()

"""### Training 

Let's *compile* the model to be able to train it. For now we'll simply use a **Mean Squared Error (MSE)** for the loss, we'll see later what this is and if we can go further.
"""

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

"""We have a dataset of images and we load it by batches.  the dataset doesn't really hold in memory, so we split it by batches and give it to the GPU so that it can train on a reasonable part of the dataset at each iteration."""

import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
import numpy as np
def train_batches(just_load_dataset=False):
    batches = 256 # Number of images to have at the same time in a batch
    batch = 0 # counter images in the current batch (grows over time and then resets for each batch)
    batch_nb = 0 # counter of current batch index
    max_batches = -1 # If you want to train only on a limited number of images to finish the training even faster.
    
    ep = 4 # Number of epochs
    images = []
    x_train_n = []
    x_train_down = []
    
    x_train_n2 = [] # Resulting high res dataset
    x_train_down2 = [] # Resulting low res dataset
    
    for root, dirnames, filenames in os.walk("/data/cars_train"): # generate the files names
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                if batch_nb == max_batches: # If we limit the number of batches, just return earlier
                    return x_train_n2, x_train_down2
                filepath = os.path.join(root, filename)
                image = pyplot.imread(filepath) # read the image file and save into an array
                if len(image.shape) > 2:
                    # Resize the image so that every image is the same size
                    image_resized = resize(image, (256, 256))
                    # Add this image to the high res dataset
                    x_train_n.append(image_resized) 
                    # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
                    x_train_down.append(rescale(rescale(image_resized, 0.5), 2.0)) 
                    batch += 1
                    if batch == batches:
                        batch_nb += 1
                        x_train_n2 = np.array(x_train_n)
                        x_train_down2 = np.array(x_train_down)
                        
                        if just_load_dataset:
                            return x_train_n2, x_train_down2
                        
                        print('Training batch', batch_nb, '(', batches, ')')
                        autoencoder.fit(x_train_down2, x_train_n2,
                            epochs=ep,
                            batch_size=10,
                            shuffle=True,
                            validation_split=0.15)
                    
                        x_train_n = []
                        x_train_down = []
                    
                        batch = 0
    return x_train_n2, x_train_down2

"""To train the model:

```Python
x_train_n = []
x_train_down = []
x_train_n, x_train_down = train_batches()
```

Training from scratch takes a lot of time, so we'll just load a pretrained model.
"""

# download CARS DATA from this url: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

#%cd /content
#!wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
#!tar -xvzf cars_train.tgz

x_train_n, x_train_down = train_batches(just_load_dataset=True)

"""And here, we load the already existing weights :"""

# Commented out IPython magic to ensure Python compatibility.
# download the WEIGHTS from a location in Dropbox
#%cd /content
!wget https://www.dropbox.com/s/n2s2n29ja5xytc7/weights.zip?dl=0 -O weights.zip
!unzip weights.zip
  
autoencoder.load_weights("/data/rez/sr.img_net.mse.final_model5.no_patch.weights.best.hdf5")

"""### Display the results

Predict:
"""

# We clip the output so that it doesn't produce weird colors
sr1 = np.clip(autoencoder.predict(x_train_down), 0.0, 1.0)

"""Display and compare the results. In the following order:

* The low-res input image
* The reconstructed image
* The original high-res image
"""

image_index = 251

import matplotlib.pyplot as plt

plt.figure(figsize=(128, 128))
i = 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index])  #Input (low-res)
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr1[image_index])  # Output (supre-res recovered image)
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n[image_index])  # Ground truth (high-res) 
plt.show()

"""## Loss Metrics

Below are the most common metrics to measure the similarity between a pair of images.

### MSE

[Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) is a metric that indicates perfect similarity if the value is 0.

The value grows beyond 0 if the average difference between pixel intensities increases.

MSE has a few issues when used for similarity comparison, meaning that if you take an image and photoshop it to make it brighter and compare it with the original, the two images will be very different according to MSE as a dark image has pixel values closer to 0 and a bright image has pixel closer to 1, this means that the difference is going to be very big as MSE sees the image from a very general point of view.

Anyway, for our purpose, we compare lower quality images to their higher resolution counterpart, so the brightness is retained, this means that ** MSE is still a useful metric in our case**.
"""

def mse(orig, res):
    return ((orig - res) ** 2).mean()

mse(high_res, low_res)

"""### SSIM

[Structural similarity (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) measures the similarity between two images as would be perceived on a television or a similar media.

The SSIM is a value in the range $[1, -1]$ where $1$ would mean two indentical images, and lower values would show a "perceptual" difference.

This metric compares small windows in the image rather than the whole image (like MSE), which makes it a bit more interesting.
"""

import skimage.measure
import os
from matplotlib import pyplot

def ssim(ori, res):
    return skimage.measure.compare_ssim(ori.astype(np.float64),
        res.astype(np.float64),
        gaussian_weights=True, data_range=1., win_size=1,
        sigma=1.5, multichannel=False, use_sample_covariance=False)

ssim(high_res, low_res)

"""### PSNR

[Peak signal-to-noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) is a metric defined using Mean Squared Error (seen above). 

*PSNR is most commonly used to measure the quality of reconstruction of lossy compression* - Wikipedia

Low resolution and pixelization can be considered as a form of *compression* as we loose information.

When there's no noise, the PSNR is infinite (because there's a division by the MSE and MSE is $0$ when both images are exactly the same).

Of course, this means that we need to maximize the PSNR. The result is in decibel (dB).
"""

def psnr(ori, res):
    return skimage.measure.compare_psnr(ori, res, data_range=1.)

psnr(high_res, low_res)

"""### HFENN

The HFENN (High Frequency Error Norm Normalized) metric gives  a  measure  of how high-frequency details differ between two images.

This means that we can guess if an image has more or less high frequency details (which are fine details that you need to zoom in to see and that are not blurry) compared to another image.

When the output value is 0 the images are identical. The greater the value, the more of a perceptual difference in both images there is.
"""

import numpy as np
from scipy.ndimage import filters

def l_o_g(img, sigma):
    '''
    Laplacian of Gaussian filter (channel-wise)
    -> img: input image
    -> sigma: gaussian_laplace sigma
    <- filtered image
    '''
    while len(img.shape) < 3:
        img = img[..., np.newaxis]
    out = img.copy()
    for chan in range(img.shape[2]):
        out[..., chan] = filters.gaussian_laplace(img[..., chan], sigma)
    return out

def hfenn(orig, res):
    '''
    High Frequency Error Norm (Normalized) metric for comparison of original and result images
    The metric independent to image size (in contrast to regular HFEN)
    Inputs are expected to be float in range [0, 1] (with possible overflow)
    -> ori: original image
    -> res: result image
    <- HFENN value
    '''
    sgima = 1.5  # From DLMRI paper
    return np.mean((l_o_g(orig - res, sgima)) ** 2) * 1e4  # magnification

hfenn(high_res, low_res)

"""## Combining Loss Functions

Let's try using a loss that doesn't just tell us the pixel-wise difference in resolution, but that also if there's an improvement, for example, in high frequency details.

### MSE and HFENN

We can do a weight sum of both losses like:

$MSE + weight * HFENN$

We could choose $weight = 10$ and see what happens.
"""

import scipy.ndimage as nd
import scipy.ndimage.filters as filters
from keras import losses
import tensorflow as tf

def hfenn_loss(ori, res):
    '''
    HFENN-based loss
    ori, res - batched images with 3 channels
    See metrics.hfenn
    '''
    fnorm = 0.325 # norm of l_o_g operator, estimated numerically
    sigma = 1.5 # parameter from HFEN metric
    truncate = 4 # default parameter from filters.gaussian_laplace
    wradius = int(truncate * sigma + 0.5)
    eye = np.zeros((2*wradius+1, 2*wradius+1), dtype=np.float32)
    eye[wradius, wradius] = 1.
    ker_mat = filters.gaussian_laplace(eye, sigma)
    with tf.name_scope('hfenn_loss'):
        chan = 3
        ker = tf.constant(np.tile(ker_mat[:, :, None, None], (1, 1, chan, 1)))
        filtered = tf.nn.depthwise_conv2d(ori - res, ker, [1, 1, 1, 1], 'VALID')
        loss = tf.reduce_mean(tf.square(filtered))
        loss = loss / (fnorm**2)
    return loss
  

def ae_loss(input_img, decoder):
    mse = losses.mean_squared_error(input_img, decoder) # MSE
    weight = 10.0 # weight
    return mse + weight * hfenn_loss(input_img, decoder) # MSE + weight * HFENN

"""Now, just compile the model with the new loss :"""

autoencoder.compile(optimizer='adadelta', loss=ae_loss)

"""If you wanted to train the model, that's how you'd do :

```Python
x_train_n = []
x_train_down = []
x_train_n, x_train_down = train_batches()
```

But, we'll just load the pretrained weights :
"""

## TODO
## download the weights

autoencoder_hfenn.load_weights("/data/rez/sr.img_net.mse.final_model5.no_patch.weights.best.hdf5")

"""Let's see what the network can do after using our new custom loss :"""

sr_hfenn = np.clip(autoencoder_hfenn.predict(x_train_down), 0.0, 1.0)

"""Display the image results:

* The low resolution input image
* A bicubic interopolated version
* The reconstructed image with MSE
* The reconstructed image with our custom MSE + HFENN loss
* The original perfect image
"""

image_index = 99

import matplotlib.pyplot as plt

plt.figure(figsize=(128, 128))
i = 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index], interpolation="bicubic")
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr1[image_index])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr_hfenn[image_index])  # The reconstructed image with our custom MSE + HFENN loss
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n[image_index])
plt.show()

"""If you look a bit closely and check out the lines and edges, you'll se that they're sharper when using MSE and HFENN compared to MSE alone."""

plt.figure(figsize=(128, 128))
j = 6
i = 1
idx_1 = 32*j
idx_2 = 32*(j+1)
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index, idx_1:idx_2, idx_1:idx_2])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index, idx_1:idx_2, idx_1:idx_2], interpolation="bicubic")
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr1[image_index, idx_1:idx_2, idx_1:idx_2])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr_hfenn[image_index, idx_1:idx_2, idx_1:idx_2])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n[image_index, idx_1:idx_2, idx_1:idx_2])
plt.show()

"""**Note** : double click on the images to make them bigger"""