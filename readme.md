
# The FX-GAN for chest-xray classification

![model](./images/GAN_model.png)
- This repository is implemented based on https://github.com/IBM/BAGAN
- To run the experiment please check the file **notebooks/fxgan.ipynb**

### Generator architecture

![G](./images/generator.png)


### Discriminator architecture

![D](./images/discriminator.png)


# Experiment results

### Data distribution of generated images (feature extracted from pre-trained metric model)
![data_dis](./images/data_dis.png)

### Generated images for 05 seen classes
![gen_seen](./images/gen_seen.png)

### Generated images for 03 unseen classes
![gen_unseen](./images/gen_unseen.png)
