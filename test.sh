ids=(proposed proposed-mn others others_best)
for id in "${ids[@]}"
do
   echo Testing "$id"
   script=test_$id.py
   docker run --runtime=nvidia -v $PWD:/home/$USER/deeprec-sib18 -it deeprec-sib18  bash -c "cd /home/$USER/deeprec-sib18 && python $script"
done
