ids=(proposed proposed-mn others others_best)
for id in "${ids[@]}"
do
   echo Testing "$id"
   script=test_$id.py
   docker run --runtime=nvidia -v $PWD:/home/$USER/sib18 -it sib18:latest  sh -c "cd /home/$USER/sib18 && python $script"
done