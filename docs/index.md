<h1 class="project-name"> SIB </h1>

# A deep learning-based compatibility score for reconstruction of strip-shredded text documents

#### [Thiago M. Paixão¹²](http://sites.google.com/site/professorpx), Rodrigo F. Berriel², Maria C. S. Boeres², Claudine Badue², Alberto F. De Souza² and Thiago Oliveira-Santos²
###### ¹Instituto Federal do Espírito Santo, ²Universidade Federal do Espírito Santo
###### paixao at gmail dot com
___

### Main dependencies:
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

___

### Reproducing the experiment:
1. Build the docker container
```
bash build.sh
```
2. Train the SqueezeNet
```
bash train.sh
```
or MobileNet model:
```
bash train-mn.sh
```
___

Besides the experiments reported on the paper, we also used the trained model to reconstruct the full collection of shredded documents after mixing their strip (aprox. 1850 strips). The resulting reconstruction is available  [here](https://daringfireball.net/projects/markdown/).
___
