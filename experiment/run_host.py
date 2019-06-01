import os








def create_host_ip_and_port(num=1,node_type=None,ip_and_port_offset=0,ip_base=0,port_bass=0):
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
    for i in range(num):

        ip_list.append('localhost:{port}'.format(ip=ip_base+i+ip_and_port_offset,port=port_bass+i+ip_and_port_offset))

    ip_string=','.join(ip_list)
    print(ip_string)

    return {node_type:ip_string}








def create_python_run_command(ps_num=1,worker_num=1,train_steps=900,ps_node_dict=None,worker_node_dict=None):
    for i in range(ps_num):
        os.system("echo 'python /home/limk/tfrecord/experiment/resnet_fullmodel_dist.py --ps_hosts {ps_list} --worker_hosts {worker_list} --job_name {job_name} --task_index {i} --train_steps {train_steps} &'>>command.sh"
              .format(ps_list=ps_node_dict['ps'], worker_list=worker_node_dict['worker'],job_name='ps',i=i,train_steps=train_steps))
        os.system("echo 'sleep 5s'>>command.sh")
    for i in range(worker_num):
        os.system("echo 'python /home/limk/tfrecord/experiment/resnet_fullmodel_dist.py --ps_hosts {ps_list} --worker_hosts {worker_list} --job_name {job_name} --task_index {i} --train_steps {train_steps} &'>>command.sh"
              .format(ps_list=ps_node_dict['ps'], worker_list=worker_node_dict['worker'],job_name='worker',i=i,train_steps=train_steps))
        if i==0:
            os.system("echo 'sleep 20s'>>command.sh")
        else:
            os.system("echo 'sleep 5s'>>command.sh")



    pass




if __name__ == '__main__':
    # os.system('set -x')
    #
    ps_num=int(input('input ps num :'))
    worker_num=int(input('input worker num :'))
    train_steps=int(input('input train_steps:'))

    # create_python_run_command(ps_node_dict={'ip_string':'1111'},worker_node_dict={'ip_string':'1111'})
    ps_ip_string=create_host_ip_and_port(ps_num,"ps",0,0,2232)
    worker_ip_string=create_host_ip_and_port(num=worker_num,node_type='worker',ip_and_port_offset=ps_num,ip_base=0,port_bass=2232)

    print(ps_ip_string,worker_ip_string)
    create_python_run_command(ps_num=ps_num,worker_num=worker_num,ps_node_dict=ps_ip_string,worker_node_dict=worker_ip_string,train_steps=train_steps)






