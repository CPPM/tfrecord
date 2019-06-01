import os
import time
import multiprocessing as mp


def get_docker_stats():
    os.system(
        r'''docker stats --no-stream --format "table {{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" |awk '{if (NR>1) {print $0}}'|awk '$2 == "limk1" {print $0}'>>a1.txt''')
    os.system(
        r'''docker stats --no-stream --format "table {{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" |awk '{if (NR>1) {print $0}}'|awk '$2 == "limk2" {print $0}'>>a2.txt''')
    os.system(
        r'''docker stats --no-stream --format "table {{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" |awk '{if (NR>1) {print $0}}'|awk '$2 == "limk3" {print $0}'>>a3.txt''')
    """
    docker stats --no-stream  --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.ID}}" |awk '{if (NR>1) {print $0}}'|awk '$1 == "16d6827ef5ee" {print $0}' 
    删除第一行  if (NR>1) {print $0}  NR代表行 $0 代表当前行 {print $0}代表输出当前行
    '$1 == "16d6827ef5ee" 匹配当前行的第一列有没有这个字符串
    
    
    显示太长可以把{{.ID}}去掉
    
    """

def get_docker_stats_2(ps_num=1,worker_num=1):
    # while True:
    # docker_command=r'''docker stats  --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" '''
    docker_command=r'''docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" '''
    awk_command=r''' |awk '{if (NR>1) {print $0}}'|awk '/limk/ {print $0}'>>monitor_a1.txt'''
    # print(docker_command+awk_command)
    test=r'''docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" >>monitor_a1.txt'''

    os.system(docker_command+awk_command)
    # os.system(test)
    print("docker2222222")
    """
    docker stats --no-stream  --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.ID}}" |awk '{if (NR>1) {print $0}}'|awk '$1 == "16d6827ef5ee" {print $0}' 
    删除第一行  if (NR>1) {print $0}  NR代表行 $0 代表当前行 {print $0}代表输出当前行
    '$1 == "16d6827ef5ee" 匹配当前行的第一列有没有这个字符串


    显示太长可以把{{.ID}}去掉

    """


def get_sar_tcp_stats():
    # os.system(r'''sar -n DEV 1 1| awk '{if (NR>3) {print $0}}'>>monitor_sar.txt''')
    os.system(r'''sar -n DEV 1 1| awk '{if (NR>3) {print $0}}'>>monitor_sar.txt''')
    print("Sar 33333333")

def get_nvidia_stats():
    # while True:
    os.system(r'''nvidia-smi --format=noheader,nounits,csv --query-gpu=memory.total,memory.used,memory.free,pstate,temperature.gpu,name,utilization.gpu,utilization.memory,index,power.draw>>monitor_gpu.csv''')
    print("GPU11111111111")
def set_column_name(s):
    os.system('''echo '{}'>>monitor_gpu.csv'''.format(s))



def get_host_stat_cpu():
    os.system(r'''pidstat -u 1 1| awk '{if (NR>2) {print $0}}'>>monitor_cpu.txt''')

def get_host_stat_mem():
    os.system(r'''pidstat -r 1 1| awk '{if (NR>2) {print $0}}'>>monitor_memory.txt''')

def get_host_stat_io():
    os.system(r'''pidstat -d 1 1| awk '{if (NR>2) {print $0}}'>>monitor_io_disk.txt''')
    print("host cpu memory io 11111111111")


def endless(fun_name):
    while True:
        fun_name()

def endless2(fun_name):
    while True:
        fun_name()

if __name__ == '__main__':



    set_column_name('memory.total,memory.used,memory.free,pstate,temperature.gpu,name,utilization.gpu,utilization.memory,index,power.draw')
    # while True:
    #     # ger_host_stat()
    #     # get_docker_stats_2()
    #     # get_sar_tcp_stats()
    #
    #     get_nvidia_stats()
    #     print("11111111111111")
    # p1 = mp.Process(target=endless(get_nvidia_stats))
    # p2 = mp.Process(target=endless(get_docker_stats_2))
    p1 = mp.Process(target=endless,args=(get_nvidia_stats,))
    p2 = mp.Process(target=endless,args=(get_docker_stats_2,))
    p3 = mp.Process(target=endless,args=(get_sar_tcp_stats,))
    p4 = mp.Process(target=endless,args=(get_host_stat_cpu,))
    p5 = mp.Process(target=endless,args=(get_host_stat_mem,))
    p6 = mp.Process(target=endless,args=(get_host_stat_io,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    print("main fun")
        # time.sleep(1)
    # while True:
    #     get_docker_stats_2()
    #     time.sleep(0.1)


    # for i in range(2):
    #     get_docker_stats_2()
    #     get_sar_tcp_stats()
    #
    #     get_nvidia_stats()

    print('end')
