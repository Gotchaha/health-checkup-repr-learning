# test_fork.py
import multiprocessing as mp
import os

def child():
    print(f"Child PID {os.getpid()} running")
if __name__ == "__main__":
    ctx = mp.get_context('fork')
    p = ctx.Process(target=child)
    p.start()
    p.join()
    print("Done")
