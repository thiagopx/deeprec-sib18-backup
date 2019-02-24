# installing container (https://docs.docker.com/install/linux/linux-postinstall/)
GPUID=0
while getopts g: option
do
case "${option}"
in
g) GPUID=${OPTARG};;
esac
done
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg GPUID=$GPUID --build-arg UNAME=$USER -t deeprec-sib18 docker
