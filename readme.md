



G =   e1                          d4 -> (d5)
        e2 ----------------> d3
            e3 ---------> d2
                e4 -> d1


x' = G(x1,x2,z)

Gloss = Fm loss + Perceptual loss + Adv loss
    + Fm loss = ||fm(x') - (fm(x1) + fm(x2)) / 2||


en = encoder

f1 = en(x1) # [1, 2, 3, 4, 5]
f2 = en(x2) # [6, 7, 8, 9, 10]

combine = f[:] + f2[:]

f = combine(f1, f2) # [1, 2, 3, 9, 10]

tìm hiểu joint 2 feature bằng một vector z kiểu phép lai

triplet loss??
