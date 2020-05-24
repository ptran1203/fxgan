motivation:
- Borrow information from k conditional image k = {0...5} to genenerate new data.
- combine feature from latent space and k conditional image.
- new image will be compare with other image from the same category

- how to determine how much information should borrow from each conditional image and the latent space?

G = Genrator()
D = Discrimninator()

k_x, x_other, x' #  k_x is k conditional images
z: random noise


x' = G(k_x, f)

D discriminate between fake distr and real distr.

D(k_x, x'): fake
D(k_x, x_other): real

We want x' is real -> x' Close to the real distribution
