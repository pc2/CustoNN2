## Researched Architecture

## AlexNet

- Contains 5 convolution layers and 3 fully connected layer (8 Layers).
- RelU activation function applied after every convolution and fully connected layer.
- First and second convolution layer is followed by overlapping max pooling layer.
- Third, fourth and fifth convolution layers are connected to each other.
- After fifth layer overlapping max pooling layer is connected.
- The output of second maxpool layer goes into a series of two fully connected layers. 
- The second fully connected layer feeds into a softmax classifier with 1000 class labels.

### Overlapping max pooling layer
Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighborhoods summarized by adjacent pooling units do not overlap (e.g., [17, 11, 4]). To be more precise, a pooling layer can be thought of as consisting of a grid of pooling units spaced s pixels apart, each summarizing a neighborhood of size z × z centered at the location
of the pooling unit. If we set s = z, we obtain traditional local pooling as commonly employed in CNNs. If we set s < z, we obtain overlapping pooling. This is what we use throughout our network, with s = 2 and z = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%, respectively, as compared with the non-overlapping scheme s = 2, z = 2, which produces output of equivalent dimensions. We generally observe during training that models with overlapping
pooling find it slightly more difficult to overfit.

### Reducing overfitting
#### Data augmentation :
- by mirroring
- by random crops

#### Drop out
In dropout, a neuron is dropped from the network with a probability of 0.5. When a neuron is dropped, it does not contribute to either forward or backward propagation. So every input goes through a different network architecture

### Source
- [Understanding Alexnet](https://www.learnopencv.com/understanding-alexnet/ "Ubderstanding Alexnet")  
- [Imagenet classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## ZFNet
- ZFNet is a variant of AlexNet but:
- CONV1: change from (11x11 stride 4) to (7x7 stride 2)
- CONV3,4,5: instead of 384, 384, 256 filters use 512, 1024, 512

### Source
- [Stanford Lecture 9](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf)


## VGGNet

- It is a variant of AlexNet but uses more layers and has smaller filters.
- VGG16 and VGG19 has 16 and 19 layers respectively.
