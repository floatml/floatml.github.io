---
layout: post
title: Paper Discussion - Learning to See by Looking at Noise
date: 2021-06-13 09:00
---

In this post, we're looking at the recent paper *[Learning to See by Looking at Noise](https://mbaradad.github.io/learning_with_noise/)* by *Manel Baradad, Jonas Wulff, Tongzhou Wang, Phillip Isola* and *Antonio Torralba*. This paper was uploaded to [arXiv](https://arxiv.org/abs/2106.05963) on Jun 10, 2021.

*(TL; DR)*

This paper explores the possibility of training deep neural networks with images generated from simple noise processes. These "noise" images are different from "noisy" images, the latter being real images with some random noise added on top. Simply put, a neural network is trained on such noise images using a contrastive loss. The goodness of the network is evaluated by using it as a fixed feature extractor and training a linear classifier on top of those features on a real dataset (ex: ImageNet) and checking its test performance. (This is the standard evaluation practice in Self Supervised learning literature). 

 ![teaser]({{ site.baseurl }}/images/llnoise_teaser.jpeg)

### Premise: Do we really need massive datasets for computer vision?
Deep neural networks typically require lots of training data to attain good performance on various computer vision tasks. Curating and labeling such huge datasets is tedious and expensive. But, what if we don't require real images for training? 

Here's the key question that the paper asks:

> What if we just encode the structural properties of real images in a noise image generation program and use the generated noise images to train neural networks?

Features of good training data for vision models: *naturalism* and *diversity*. This paper aims to encode some structural properties of natural images into simple noise models. Such noise models are used to generate thousands of *naturalistic* training images without any human effort. 

### Generative Image Models (Without Training!)
At a very high level, sampling from generative models can be thought of as sampling from the following joint distribution:  

![images]({{ site.baseurl }}/images/llnoise_eqn.png)

For generative models such as trained GANs, the parameters $\theta$ are fit to real data. In our case, the parameters are sampled from simple mathematical prior distributions. The following is the list of the generative models used in this paper:

 1. Procedural Image Models: FractalDB, CLEVR, DMLab, MineRL
 2. Dead Leaves: Squares, Oriented, Shapes, Textured
 3. Statistical Models: Spectrum, Wavelet-marginal model (WMM), Color Histograms
 4. Style GAN: randomly initialized, high-freq, sparse, oriented
 5. Feature Visualisation: Random, Dead Leaves

The below figure shows the examples of generated images from each method.

 ![images]({{ site.baseurl }}/images/llnoise_images.png)

The details of the methods are omitted for the sake of brevity, however, I'll list out the key assumptions used in sampling noise images using some of the above methods:

 1. The magnitude of the Fourier Transform of natural images generally follows a [power law](https://en.wikipedia.org/wiki/Power_law).
 2. Wavelet coefficients of natural images follow a non-gaussian distribution. 
 3. Natural images have a high degree of sparsity.

Section 3 in the [paper](https://arxiv.org/pdf/2106.05963.pdf) mentions the details of each sampling method.
### Main Experiment
For each proposed image generation method, 105k samples are generated. They're used to train an AlexNet based encoder using a contrastive loss known as Alignment and Uniformity loss. This loss basically ensures that (1) the positive pairs are closer in the feature space and (2) the feature vectors should be distributed uniformly on a unit hypersphere. 

Once the training is done, the goodness of the model is evaluated on real image datasets by training a linear classifier on top of the trained encoder.

![results]({{ site.baseurl }}/images/llnoise_result.png)

VTAB (Visual Task Adaptation Benchmark) consists of 19 vision tasks divided into three categories: 

 - Natural - images taken by consumer cameras
 - Specialized  - Ex: medical images, satellite captured images
 - Structured - tasks which need understanding of the scene structure, Ex: object counting, depth estimation

The above figure shows the average performance of each method on all the tasks in each of the three categories. The black bars are baselines: the ones on the left being untrained methods, and the ones on the right being models trained on real datasets - Places, ImageNet-100, ImageNet-1k. Here are the main observations:

**Natural Images (left most plot)**
 1. Obviously, the images trained on real datasets (black bars at the left) attain the highest performance.
 2. The performance of the image generation methods seems correlated with the naturalism and diversity of the generated images. (Ex: StyleGAN generated images have higher accuracy)

**Specialized (middle plot) and Structured images (right most plot)**

 1. Interestingly, the generated images attain performance pretty much close to the real image datasets. This means that the specialized and structured tasks don't benefit much from real images. 
 2. More interestingly, even a randomly initialized untrained CNN (third bar in the plot) achieves comparable performance with the models trained on real datasets. 

### Visualizing Features 

![visualise]({{ site.baseurl }}/images/llnoise_visualise.png)

Are the features learned from generated images different from the features learned from real images? 
Turns out, they don't look exactly similar but they capture sufficient structural info (edges etc.) just like the real image features. 

### What makes a good generated dataset?
The paper lists out some image level and dataset level properties which make a good generated datasets i.e., which give good downstream task performance. Feel free to check out the original paper for the experiments behind these conclusions:

**At the image level**
 1. It helps to have a color distribution close to the real image datasets.
 2. The frequency spectrum of the generated images should be close to the natural images.
 3. An augmentation of a generated image should be similar to the generated image and dissimilar from other images in the dataset. This aids contrastive learning. 

**At the dataset level:**

 1. The generated image distribution should be closer to the test set distribution of the real image dataset.
 2. Diversity is important. In other words, a highly naturalistic generated image dataset which is concentrated on a small portion of the test set distribution doesn't get good performance because the representations won't be diverse enough. 

### My Thoughts
This paper shows interesting ways of generating noise images which help generated useful pretrained models for vision tasks. Also, the promising results in this paper remind me of an age-old question: "What do CNNs actually learn?". If the CNN architecture and noise images together are enough to give a good naturalistic prior, then why bother with labeling millions of real image examples? I hope this field sees more improvements in learning from noise, thus reducing lots of labeling effort.

During the experiments, it is reported that 105k images are generated using each noise image generation method, which is still a huge number. While manual effort isn't needed to generate the images, there are still memory requirements (to store the images) and possibly high training time (GPU-hours). If one wants to improve the methods, this seems to be a good  place to start.

It would also be worthwhile to generate visual explanations (ex: Grad CAM) for models trained on noise images, and compare with models trained on real images. 

The paper uses 20k training examples at max, to train the linear classifiers on top of the alexnet encoders. Since the aim eventually is to do away with most of the real data, it would be interesting to see the performance of the proposed methods on VTAB-1k (only 1000 training examples per task).

### Further Readings
1. [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) - Using untrained deep networks for inverse image tasks.
2. [ Pre-training without Natural Images](https://arxiv.org/abs/2101.08515) - Using fractals to automatically generate images for pretraining vision models.
3. [The Wavelet Transform](https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34)
4. [Statistics of Natural Image Categories (mit.edu)](http://web.mit.edu/torralba/www/ne3302.pdf)

This post is just a summary of the paper. I would strongly encourage you to check out their website and read the original paper for details and more interesting results. *[Discuss this post on Twitter](https://twitter.com/floatml/status/1403995141807906817)*.

*Written on 13 June, 2021.*
