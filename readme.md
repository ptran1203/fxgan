1. Apply Unet and resNet for Generator like suggested architechure in DAGAN and Few-shot face genreration
2. Input is the real images (from specific class), go througt a Decoder into latent vector, and then merge that vector with **latent space** of this class
3. **latent space** is learned via another decoder
4. Use L1 loss combine with feature matching loss, perceptual loss for Generator


5. lấy latent l1 trừ or chia image l1, nếu latent giống nhau -> image giống
                                       nếu latent khác -> image phải khác