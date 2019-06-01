import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df =pd.read_csv(open(r'monitor_gpu.csv'))
print(df.head())
# print(df['index']==0)


def count_gpu_utilization_ratio(df=None,gpu_num=0,ratio=0):

    for  i in range(gpu_num):

        a=df[(df['index']==i)&(df['utilization.gpu']>ratio)]['utilization.gpu']
        b=df[(df['index']==i)]['utilization.gpu']


        # print(a.shape)
        # print(b.shape)
        print("GPU {}利用率{}%以上时间占比 {}".format(i,ratio,a.shape[0]/b.shape[0]))

count_gpu_utilization_ratio(df,gpu_num=4,ratio=50)





def show_gpu_utilization(df=None,gpu_num=0):
    plt.figure()
    for i in range(gpu_num):
        # print(df[(df['index']==i)]['utilization.gpu'])

        a = df[(df['index'] == i)]['utilization.gpu']
        plt.subplot(1, gpu_num, i + 1)
        plt.plot(a)
    plt.xlabel('time  s')
    plt.ylabel('gpu_utilization  %')
    plt.show()


show_gpu_utilization(df,gpu_num=4)



# plt.figure()
#
#
# for i in range(2):
#     # print(df[(df['index']==i)]['utilization.gpu'])
#
#     a=df[(df['index']==i)]['utilization.gpu']
#     plt.subplot(1,2,i+1)
#     plt.plot(a)
# # print(a)
# # print(type(a))
#
#
#
# plt.show()