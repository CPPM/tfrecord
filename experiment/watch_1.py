import os
import re
import time
import logging

file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'nvidia-smi-{}.log'.format(str(int(time.time()))))
logging.basicConfig(filename=file_name, level=logging.INFO)


## 收集 cpu 内存 gpu的利用率



def watch_cpu():
    pid_str = os.popen('ps -aux|grep /home/limk/tfrecord/experiment/resnet_fullmodel_dist.py').read()
    pids = re.findall(r'(root.*?)python\s/home/limk/tfrecord/experiment/resnet_fullmodel_dist.py',
                      pid_str, re.S)
    #
    print(pid_str)
    print(pids)
    # print(len(pids))
    if pids:
        pid = pids[0]
        return pid
    else:
        return ''

#


if __name__ == '__main__':
    watch_cpu()

