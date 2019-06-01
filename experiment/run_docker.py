import os


def create_ps(num=1):
    for i in range(num):
        os.system(r'''docker run -d -it --name limk-ps-{} \
                     --network limk-network \
                        --ip 172.172.0.{} \
                        -p 223{}:223{} \
                       -v /home/limk:/home/limk \
                     -v /data/train:/data/train \
                     --privileged \
                    10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash 
        '''.format(i, i + 2, i + 2, i + 2))

    pass


def print_test(num=1,node_type=None,ip_and_port_offset=0,ip_base=0,port_bass=0):
    """

    :param num:
    :param node_type:
    :param ip_and_port_offset:
    :param ip_base:
    :param port_bass:
    :return:一个ip和端口的字典 dict={node_type:"ip:port,ip:port"}
            和 {'node_name_string': 'limk-worker-0,limk-worker-1,}

    """


    ip_list=[]
    node_name_list=[]
    for i in range(num):
        print(r'''docker run -d -it --name limk-{node_type}-{node_rank} \
                     --network limk-network \
                        --ip 172.172.0.{ip} \
                        -p {port}:{port} \
                       -v /home/limk:/home/limk \
                     -v /data/train:/data/train \
                     --privileged \
                    10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash 
        '''.format(node_type=node_type,node_rank=i,ip=ip_base+i+ip_and_port_offset,port=port_bass+i+ip_and_port_offset))
        ip_list.append('172.172.0.{ip}:{port}'.format(ip=ip_base+i+ip_and_port_offset,port=port_bass+i+ip_and_port_offset))
        node_name_list.append('limk-{node_type}-{node_rank}'.format(node_type=node_type,node_rank=i))
    ip_string=','.join(ip_list)
    node_name_string=','.join(node_name_list)
    return {node_type:ip_string,'node_name_string':node_name_string}


def create_container(num=1,node_type=None,ip_and_port_offset=0,ip_base=0,port_bass=0):
    """

    :param num:
    :param node_type:
    :param ip_and_port_offset:
    :param ip_base:
    :param port_bass:
    :return:一个ip和端口的字典 dict={node_type:"ip:port,ip:port"}
            和 {'node_name_string': 'limk-worker-0,limk-worker-1,}

    """


    ip_list=[]
    node_name_list=[]
    for i in range(num):
        os.system(r'''docker run -d -it --name limk-{node_type}-{node_rank} \
                     --network limk-network \
                        --ip 172.172.0.{ip} \
                        -p {port}:{port} \
                       -v /home/limk:/home/limk \
                     -v /data/train:/data/train \
                     --privileged \
                    10.59.3.203:5000/tensorflow1.13_cuda10_py3   /bin/bash 
        '''.format(node_type=node_type,node_rank=i,ip=ip_base+i+ip_and_port_offset,port=port_bass+i+ip_and_port_offset))
        ip_list.append('172.172.0.{ip}:{port}'.format(ip=ip_base+i+ip_and_port_offset,port=port_bass+i+ip_and_port_offset))
        node_name_list.append('limk-{node_type}-{node_rank}'.format(node_type=node_type,node_rank=i))
    ip_string=','.join(ip_list)
    node_name_string=','.join(node_name_list)
    return {node_type:ip_string,'node_name_string':node_name_string}






def create_delete_sh_file(*args):
    for arg in args:
        if 'node_name_string' not in arg:
            raise ValueError('no node_name_string key')
        node_name_list=arg['node_name_string'].split(',')
        for node_name in node_name_list:

            os.system("echo 'docker rm -f {}'>>delete_containor.sh".format(node_name))
        # [arg['node_name_string'].spilt(',')]

def create_python_run_command(ps_num=1,worker_num=1,train_steps=900,ps_node_dict=None,worker_node_dict=None):
    for i in range(ps_num):
        os.system("echo 'docker exec -d limk-{job_name}-{i}   python /home/limk/tfrecord/experiment/resnet_fullmodel_dist.py --ps_hosts {ps_list} --worker_hosts {worker_list} --job_name {job_name} --task_index {i} --train_steps {train_steps} '>>command.sh"
              .format(ps_list=ps_node_dict['ps'], worker_list=worker_node_dict['worker'],job_name='ps',i=i,train_steps=train_steps))
    for i in range(worker_num):
        os.system("echo 'docker exec -d limk-{job_name}-{i}   python /home/limk/tfrecord/experiment/resnet_fullmodel_dist.py --ps_hosts {ps_list} --worker_hosts {worker_list} --job_name {job_name} --task_index {i} --train_steps {train_steps} '>>command.sh"
              .format(ps_list=ps_node_dict['ps'], worker_list=worker_node_dict['worker'],job_name='worker',i=i,train_steps=train_steps))



    pass





def create_ps_and_worker(ps_num=1,worker_num=1,train_steps=900):
    ip_and_port_offset=ps_num
    ip_base=2
    port_bass=2232
    ps_node_list=create_container(ps_num,node_type='ps',ip_and_port_offset=0,ip_base=ip_base,port_bass=port_bass)
    print(ps_node_list)
    worker_node_list=create_container(worker_num,node_type='worker',ip_and_port_offset=ip_and_port_offset,ip_base=ip_base,port_bass=port_bass)
    print(worker_node_list)
    create_delete_sh_file(ps_node_list,worker_node_list)
    create_python_run_command(ps_num=ps_num,worker_num=ps_num,train_steps=train_steps,ps_node_dict=ps_node_list,worker_node_dict=worker_node_list)





    pass

def record_veth_and_container():
    os.system('sh vethfinder.sh>>container_and_veth.csv')
    print("record_veth_and_container")


if __name__ == '__main__':
    os.system('set -x')

    ps_num=int(input('input ps num :'))
    worker_num=int(input('input worker num :'))
    train_steps=int(input('input train_steps:'))
    create_ps_and_worker(ps_num=ps_num, worker_num=worker_num,train_steps=train_steps)
    record_veth_and_container()
    # create_python_run_command(ps_node_dict={'ip_string':'1111'},worker_node_dict={'ip_string':'1111'})




