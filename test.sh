ids=(proposed) # proposed-mn others others_best)
for id in "${ids[@]}"
do
   echo Testing "$id"
   script=test_$id.py
   docker run --runtime=nvidia --user=$USER -v $PWD:/home/$USER/deeprec-sib18 -v /opt/localsolver:/opt/localsolver:ro -it deeprec-sib18  bash -c "cd /home/$USER/deeprec-sib18 && python3 $script"
done
