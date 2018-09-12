modules=(accuracy accuracy_best accuracy_proposed)
for module in "${modules[@]}"
do
   echo Generating graph "$module"
   docker run -v $PWD:/home/$USER/sib18 -it sib18:latest  sh -c "cd /home/$USER/sib18 && python -m graphs.$module"
done