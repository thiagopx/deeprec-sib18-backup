docker run --runtime=nvidia --user=$USER -v $PWD:/home/$USER/deeprec-sib18 -it deeprec-sib18 bash -c "cd /home/$USER/deeprec-sib18 && python3 dataset.py"
