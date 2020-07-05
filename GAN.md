# Generative models

In 2014, Goodfellow et al. (2014) introduced Generative Adversarial Networks (GANs). They are generative models with the objective of learning the underlying distribution of training data in order to generate new realistic data samples which are indistinguishable from the input dataset. Prior to the introduction of GANs, state-of-the-art generation models, such as Variational Autoencoders (VAE) (Kingma and Welling, 2013, Rezende et al., 2014), tackled this task by performing explicit density estimation. GANs constitute an alternative to this by defining a high-level goal such as “generate output data samples which are indistinguishable from input data” and minimizing the loss function through a second adversarial network instead of explicitly defining it.

## GAN Architecture(follows game theory)- GENERATOR(ENCODER+DECODER)-generates candidates & DISCRIMINATOR-evaluates candidates

The main underlying principle of GANs is that of rivalry and competition between two co-existing networks. The first network, the generator, takes random noise as input and outputs synthetic data samples. The second network, the discriminator, acts as a binary classifier which attempts to distinguish between real training data samples and fake synthetic samples from the generator. In the training procedure, the two networks are trained simultaneously with opposing goals. The generator is instructed to maximize the probability of fooling the discriminator into thinking the synthetic data samples are realistic. On the other hand, the discriminator is trained to minimize the cross entropy loss between real and generated samples, thus maximize the probability of correctly classifying real and synthetic images.
Convergence is achieved by GANs from a game theory point of view by reaching Nash equilibrium (Zhao et al., 2016). Thus, the distribution of the generator network will converge to that of the training data and the discriminator will be maximally confused in distinguishing between real and fake data samples.
The GAN architecture is comprised of a generator model for outputting new plausible synthetic images, and a discriminator model that classifies images as real (from the dataset) or fake (generated). The discriminator model is updated directly, whereas the generator model is updated via the discriminator model. As such, the two models are trained simultaneously in an adversarial process where the generator seeks to better fool the discriminator and the discriminator seeks to better identify the counterfeit images. This logical or composite model involves stacking the generator on top of the discriminator. A source image is provided as input to the generator and to the discriminator, although the output of the generator is connected to the discriminator as the corresponding “target” image. The discriminator then predicts the likelihood that the generator was a real translation of the source image.
The generator is trained via adversarial loss, which encourages the generator to generate plausible images in the target domain. The generator is also updated via L1 loss measured between the generated image and the expected output image. This additional loss encourages the generator model to create plausible translations of the source image. The encoder and decoder of the generator are comprised of standardized blocks of convolutional, batch normalization, dropout, and activation layers. This standardization means that we can develop helper functions to create each block of layers and call it repeatedly to build-up the encoder and decoder parts of the model.

## GAN Applications:
Image super resolution
Image to image translation
Style transfer
Text to Image generation

## Examples:
Generate Examples for Image Datasets
Generate Photographs of Human Faces
Generate Realistic Photographs
Generate Cartoon Characters
Image-to-Image Translation
Text-to-Image Translation
Semantic-Image-to-Photo Translation
Face Frontal View Generation
Generate New Human Poses
Photos to Emojis
Photograph Editing
Face Aging
Photo Blending
Super Resolution
Photo Inpainting
Clothing Translation
Video Prediction
3D Object Generation

Ref: https://arxiv.org/pdf/1611.07004.pdf
