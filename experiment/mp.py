import multiprocessing as mp


def f1():
    while True:
        print('111111111')

def f2():
    while True:
        print('222222222')

if __name__ == '__main__':



    p1 = mp.Process(target=f1())
    p2 = mp.Process(target=f2())
    p1.start()
    p2.start()