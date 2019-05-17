#! /bin/bash -f	
#--runtime=
if [ "$1" = "run" ];then
	docker run --runtime=nvidia -it --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -e duid=`id -u` \
	-e dgid=`id -g` -e username=$USER \
	-e cur_path=`pwd` \
	-v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home -v /mnt:/mnt \
	--hostname=docker_chensy \
	-w $PWD --user $2 --shm-size 4G \
	-p 5002:6006 \
	pytorch_chensy2:base
# user can either be 'root' or '$(id -u)'
elif [ "$1" = "commit" ]
then
	docker commit $(docker ps -l -q) pytorch_chensy2:base
fi

