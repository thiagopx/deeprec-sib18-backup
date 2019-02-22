docker run --runtime=nvidia -v $PWD:/home/$USER/deeprec-sib18 -it deeprec-sib18:latest bash -c ". envs/deeprec-sib18/bin/activate && cd /home/$USER/deeprec-sib18 && python dataset.py"
