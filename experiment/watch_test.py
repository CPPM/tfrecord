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
def watch_gpu():
    while True:
        nvidia_smi = os.popen('nvidia-smi').read()
        items = re.findall(r'(\d+)\s+Tesla.*?(\d+)MiB.*?(\d+)MiB.*?(\d+)%',
                           nvidia_smi, re.S)
        # print(nvidia_smi)
        print(items)
        cpu_info = watch_cpu()
        if cpu_info:
            cpu_pid = re.findall('root\s+(\d+)\s', cpu_info, re.S)[0]
            cpu_index = os.popen('ps -p {} -o lastcpu'.format(cpu_pid)).read()
            cpu_index = re.findall('\d+', cpu_index, re.S)[0]
        else:
            cpu_index = -1
        logging.info('gpu:{},  cpu:{}, cpu_index:{}'.format(' '.join([' '.join(i) for i in items]),
                                                            cpu_info,
                                                            cpu_index))
        time.sleep(1)


if __name__ == '__main__':
    # watch_cpu()
    watch_gpu()
