import multiprocessing as mp
import time

def f1():
    # while True:
    for i in range(10):
        time.sleep(1)
        print('111111111')

def f2():
    # while True:
    for i in range(10):
        time.sleep(1)

        print('222222222')

if __name__ == '__main__':



    p1 = mp.Process(target=f1)
    p2 = mp.Process(target=f2)
    p1.start()
    p2.start()
    p2.join()
    p1.join()