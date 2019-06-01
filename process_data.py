import os
# a=os.popen(r"cat container_and_veth.csv |awk -F , '/limk/ {print $2}'")
# print(a.read())
# print(type(a.read()))
veth_and_containor=[]
command=r"cat container_and_veth.csv |awk -F , '/limk/ {print $2,$3}'"
os.system(command)
with os.popen(command, "r") as p:


    for line in p.readlines():
        line = line.strip('\n')
        veth_and_containor.append(line)

for i in veth_and_containor:
    a=i.split(" ")
    print(a)
    c1=r'''cat monitor_sar.txt | awk '/{veth}/ '''.format(veth=a[0])
    c2=r'''{print $0}' | awk '/Average: / {print $0}' >>'''
    c3=r'''{containor}.txt'''.format(containor=a[1])
    c_all=c1+c2+c3
    print(c_all)
    os.system(c_all)