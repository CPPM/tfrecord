import os
a=os.popen(r"cat container_and_veth.csv |awk -F , '/limk/ {print $2}'")
print(a.read())
print(type(a.read()))
print(a.read().split(' '))