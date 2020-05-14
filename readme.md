1. Apply Unet and resNet for Generator like suggested architechure in DAGAN and Few-shot face genreration
2. Input is the real images (from specific class), go througt a Decoder into latent vector, and then merge that vector with **latent space** of this class
3. **latent space** is learned via another decoder
4. Use L1 loss combine with feature matching loss, perceptual loss for Generator


5. lấy latent l1 trừ or chia image l1, nếu latent giống nhau -> image giống
                                       nếu latent khác -> image phải khác




z, g(z) }                    } adversarial loss
        }   Discriminator -> 
e(x), x }                    } Similiarity loss: l1(z, e(x)) / l1(g(z), x)


* l1(z, e(x)) / l1(g(z), x)

if l1(z, e(x)) small -> 


|x_fake1 - x_fake2| - |x_real1 - x_real2|

gioi thieu bai toan
thach thuc
uu nhuoc diem pp hien co
phuong phap de xuat
ket luan
