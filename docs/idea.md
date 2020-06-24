
The idea:
 Force the image X to have the similiar feature to conditional image Y by normalize feature of image input X to have similiar distribution to image Y

Current works used Feature normalization to normalize the feature distribution of a layer close to the distribution of 'Discriminative feature' extracted from pre-trained metric model.
- **Formulation**

![equation](../images/feature_norm.png)  



## Model options:
1. **sampling from real images**
```
require:
    x: image
    n = 2 # (n shot)
    G = Generator
    D = Discriminator
    F = pre-trained VGG16 on i

# training G and D
While model not converge do:
    # train G #
    draw x from dataset
    f = F(x)
    z ~ p(0, 1)
    x' = G(f, z) # concatenate f and z
    loss_G = loss_adv + λ*l2(f - F(x')) + γ*l2(x - x')
    update weights for G with loss_G

    # train D #
    x = real_images
    x' = fake_images
    loss_D = hinge(x, x')
    update weights for D with loss_D
```

2. **sampling from latent z**
```
require:
    x: image
    n = 2 # (n shot)
    G = Generator
    D = Discriminator
    F = pre-trained VGG16 on i

# compute multivariate distribution #
z ~ p(class_id)

# training G and D
While model not converge do:
    # train G #
    draw x from dataset
    z ~ p(class_id)
    f = F(x)
    x' = G(f, z) # concatenate f and z
    loss_G = loss_adv + λ*l2(f - F(x')) + γ*l2(x - x')
    update weights for G with loss_G

    # train D #
    x = real_images
    x' = fake_images
    loss_D = hinge(x, x')
    update weights for D with loss_D
```

## Training note
https://arxiv.org/pdf/1612.00005.pdf
To stabilize the GAN training, we follow heuristic rules
based on the ratio of the discriminator loss over generator
loss r = lossD/lossG and pause the training of the generator or discriminator if one of them is winning too much. In
most cases, the heuristics are

a) pause training D if r < 0.1;
b) pause training G if r > 10.


## Paper note
https://arxiv.org/pdf/1612.00005.pdf
In this paper, we propose a class of models called PPGNs that are composed of
1) a generator network G that is trained to draw a wide range of image types
2) a replaceable “condition” network C that tells G what to draw Panel (b) and (c) show the components involved in the training of the generator network G for two main PPGN variants
experimented in this paper. Only shaded components (G and D) are being trained while others are kept frozen. b) For
the Noiseless Joint PPGN-h variant (Sec. 3.5), we train a generator G to reconstruct images x from compressed features h
produced by a pre-trained encoder network E. Specifically, h and h1 are, respectively, features extracted at layer fc6 and
pool5 of AlexNet [26] trained to classify ImageNet images (a). G is trained with 3 losses: an image reconstruction loss
Limg, a feature matching loss [9] Lh1
and an adversarial loss [14] LGAN . As in Goodfellow et al. [14], D is trained to tell
apart real and fake images. This PPGN variant produces the best image quality and thus used for the main experiments in
this paper (Sec. 4). After G is trained, we sample from this model following an iterative sampling procedure described in
Sec. 3.5. c) For the Joint PPGN-h variant (Sec. 3.4), we train the entire model as being composed of 3 interleaved DAEs
respectively for x, h1 and h. In other words, we add noise to each of these variables and train the corresponding AE with a
L2 reconstruction loss. The loss for D remains the same as in (a), while the loss for G is now composed of 4 components:
L = Limg + Lh1 + Lh + LGAN . The sampling procedure for this PPGN variant is provided in Sec. 3.4. See Sec. S9 for
more training and architecture details of the two PPGN variants.
