## A Deep Learning-Based Compatibility Score for Reconstruction of Strip-Shredded Text Documents

[Thiago M. Paixão](https://sites.google.com/site/professorpx), [Rodrigo F. Berriel](http://rodrigoberriel.com), [Maria C. S. Boeres](http://www.inf.ufes.br/~boeres), [Claudine Badue](https://www.inf.ufes.br/~claudine/), [Alberto F. De Souza](https://inf.ufes.br/~alberto), and [Thiago Oliveira-Santos](https://www.inf.ufes.br/~todsantos/home)

Paper presented in the 31st Conference on Graphics, Patterns and Images (SIBGRAPI 2018). The manuscript is available at the [IEEExplore](https://ieeexplore.ieee.org/abstract/document/8614315) platform and at [SIBGRAPI](http://sibgrapi.sid.inpe.br/rep/sid.inpe.br/sibgrapi/2018/09.03.21.36?metadatarepository=sid.inpe.br/sibgrapi/2018/09.03.21.36.30&ibiurl.backgroundlanguage=en&ibiurl.requiredsite=sibgrapi.sid.inpe.br+802&requiredmirror=sid.inpe.br/banon/2001/03.30.15.38.24&searchsite=sibgrapi.sid.inpe.br:80&searchmirror=sid.inpe.br/banon/2001/03.30.15.38.24) Digital Library Archive.

### BibTeX
```
@inproceedings{paixao2018deep,
  title={A deep learning-based compatibility score for reconstruction of strip-shredded text documents},
  author={Paixao, Thiago M and Berriel, Rodrigo F and Boeres, Maria CS and Badue, Claudine and De Souza, Alberto F and Oliveira-Santos, Thiago},
  booktitle={2018 31st SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  pages={87--94},
  year={2018},
  organization={IEEE}
}
```

#### Abstract

The use of paper-shredder machines (mechanical shredding) to destroy documents can be illicitly motivated when the purpose is hiding evidence of fraud and other sorts of crimes. Therefore, reconstructing such documents is of great value for forensic investigation, but it is admittedly a stressful and time-consuming task for humans. To address this challenge, several computational techniques have been proposed in literature, particularly for documents with text-based content. In this context, a critical challenge for automated reconstruction is to measure properly the fitting (compatibility) between paper shreds (strips), which has been observed to be the main limitation of literature on this topic. The main contribution of this paper is a deep learning-based compatibility score to be applied in the reconstruction of strip-shredded text documents. Since there is no abundance of real-shredded data, we propose a training scheme based on digital simulated-shredding of documents from a well-known OCR database. The proposed score was coupled to a black-box optimization tool, and the resulting system achieved an average accuracy of 94.58% in the reconstruction of mechanically-shredded documents.

---

### Reproducing the experiments
In construction.


The experiments can be reproduced with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) container technology. After installing Docker in our environment, make sure you are able to run Docker containers as non-root user (check this [guide](https://docs.docker.com/install/linux/linux-postinstall) for additional information). Then, run the following bash commands in a terminal:

1. Clone the project repository and enter the project directory:
```
git clone https://github.com/thiagopx/deeprec-sib18.git
cd deeprec-sib18
```

2. Build the container (defined in ```docker/Dockerfile```):
```
bash build.sh
```
The container includes all dependencies except the [LocalSolver](https://www.localsolver.com/) optimizer, which requires a license. You should install LocalSolver locally in ```/opt/localsolver```.

3. Generate the training samples:
```
bash dataset.sh
```

4. Train the models:
```
bash train.sh    # for SqueezeNet
bash train-mn.sh # for MobileNet
```

5. Run the experiment (includes all methods/architectures):
```
bash test.sh
```
The results will be placed at ```results``` directory.

**Important**: Note that ```test.sh``` creates a volume from the local ```/opt/localsolver``` directory.

6. Generate the graphs:
```
bash graphs.sh
```
By default, the graphs will be placed at ```graphs``` directory.
