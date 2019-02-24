docker run --runtime=nvidia --user=$USER -v $PWD:/home/$USER/deeprec-sib18 -it deeprec-sib18  bash -c "cd /home/$USER/deeprec-sib18 && python3 train.py -e 5 -bs 128 -lr 0.0001 -s 0.33 -d datasets/patches"



