# installing container (https://docs.docker.com/install/linux/linux-postinstall/)
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$USER -t sib18:latest docker