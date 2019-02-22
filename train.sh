docker run --runtime=nvidia --user=$USER -v $PWD:/home/$USER/deeprec-sib18 -it deeprec-sib18:latest  sh -c ". envs/deeprec-sib18/bin/activate && cd /home/$USER/deeprec-sib18 && python train.py -e 5 -bs 128 -lr 0.0001 -s 0.33 -d datasets/patches"



