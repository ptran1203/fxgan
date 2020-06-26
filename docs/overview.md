# Overview of existing methods


## 1. MatchingGAN
- base idea: Borrow information from N images in the same class and use matching procedure to generate new images
- The information in the same class keep unchanged by auxiliary classifier in Discriminator

## 2. DAGAN
- Borrow information from single image, apply transformation by injecting noise to the latent space

## 3. OpenGAN
- Generate image from prior distribution (latent space) and inject the discriminative feature by feature normlization extracted from a pre-trained metric embbeding.
