docker run -d -it --name limk1 \
--network limk-network \
--ip 172.172.0.2 \
-p 2232:2232 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk2 \
--network limk-network \
--ip 172.172.0.3 \
-p 2233:2233 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk3 \
--network limk-network \
--ip 172.172.0.4 \
-p 2234:2234 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk4 \
--network limk-network \
--ip 172.172.0.5 \
-p 2235:2235 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk5 \
--network limk-network \
--ip 172.172.0.2 \
-p 2232:2232 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk6 \
--network limk-network \
--ip 172.172.0.2 \
-p 2232:2232 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk7 \
--network limk-network \
--ip 172.172.0.2 \
-p 2232:2232 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
docker run -d -it --name limk8 \
--network limk-network \
--ip 172.172.0.2 \
-p 2232:2232 \
-v /home/limk:/home/limk \
-v /data/train:/data/train \
--privileged \
10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash
