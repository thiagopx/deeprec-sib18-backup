docker run --runtime=nvidia -v $PWD:/home/$USER/sib18 -it sib18:latest  sh -c "cd /home/$USER/sib18 && python train.py -e 5 -bs 128 -lr 0.0001 -s 0.33 -d datasets/patches"



