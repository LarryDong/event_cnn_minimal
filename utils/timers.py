import torch
import time
import numpy as np
import atexit
from collections import defaultdict

cuda_timers = defaultdict(list)
timers = defaultdict(list)


class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    # __enter__和__exit__方法，即支持上下文管理器协议。上下文管理器就是支持上下文管理器协议的对象，它是为了with而生。当with语句在开始运行时，会在上下文管理器对象上调用 __enter__ 方法。with语句运行结束后，会在上下文管理器对象上调用 __exit__ 方法
    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()    # Waits for all kernels in all streams on a CUDA device to complete
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))


class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        timing_value = np.mean(np.array(timing_values))
        if timing_value < 1000.0:
            print('{}: {:.2f} ms ({} samples)'.format(timer_name, timing_value, len(timing_values)))
        else:
            print('{}: {:.2f} s ({} samples)'.format(timer_name, timing_value / 1000.0, len(timing_values)))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
