"""This will have the code that monitors performance and have some optimization codes

- Use Pytorch data loader with pinned memory (pin_memory) to see if you gain any performance increase. 
You might not notice significant improvements since the dataset is small.

- Build in cython:

from Cython.Build import cythonize

setup(
    ext_modules = cythonize("extensions.pyx")
)

- After getting the unit tests to work you know the function works, then add a timeit wrapper and time 
those functions and then add a logger wrapper that saves those timing functions in an excel file to
be able to be used later.

- I can go into train, eval and interpret and add timing calls to start timing and then save it to a log
for the log, I think I can write a function but for timing it will have to be written inside functions
which will mess how the code looks, can I have a simple function that calls things from here.

"""

import cProfile as profiles
import inspect
import linecache
import os
import pstats
import sys
import time
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from threading import Thread

import psutil

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "pytorch_unet.trainer"

from pytorch_unet.trainer.train import main
from pytorch_unet.utils.helpers import elapsed_since, format_bytes


def profile_time(function, *args, **kwargs):
    """Profiles ncall, tottime percall cumtime and percall for the top 20 slow parts of the program.
    Note:
        Start by _run which runs the function by calling the modules function and calling it as __profile_run__ 
        then func_id gets the name of the module and profile starts profiling the times using cProfile, 
        then passing the func_id to pstats which reads the files into a single object and stream streams 
        the function to profile_time dump which is then again opened and converted to log format.
    :param function     : Pass in the function to be time profiled.
    :return (string)    : Timing profile.
    """

    def _run():
        function(*args, **kwargs)

    sys.modules['__main__'].__profile_run__ = _run
    func_id = function.__name__ + '()'
    profiles.run('__profile_run__()', func_id)

    p = pstats.Stats(func_id)
    p.stream = open(func_id, 'w')
    p.dump_stats('./profile_time.dmp')
    p.stream.close()
    s = open(func_id).read()
    os.remove(func_id)

    out_stream = open('./profile_time.log', 'w')
    ps = pstats.Stats('./profile_time.dmp', stream=out_stream)
    ps.strip_dirs().sort_stats('time').print_stats(20)
    print("Time Profiling Complete!")

    return s


def get_process_memory():
    """Function to get process memory using psutil.
    Note:
        The os.getpid is usd to get the process identification number (it is the number automatically assigned
        to each process). Then memory_info gets a bunch of memory information. 
    :return: 
        RSS (Resident Set Size)         : the non-swapped physical memory a process has used.
        VMS (Virtual Memory Size)       : the total amount of virtual memory used.
        num_page_faults                 : Memory that could be potentially shared with other processes. 
    """

    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.num_page_faults


def display_top(snapshot, key_type='lineno', limit=20):
    """Function to display the top traces.
    Note:
        Start by ignoring <frozen importlib._bootstrap> and <unknown> files and using statistics group by line no.
        Then we enumerate the top_stats and get the frame, filename, line number, line name and the RSS bytes
        for each of the top 20 traces.
        Based on: https://pytracemalloc.readthedocs.io/examples.html.
    :param snapshot         : A snapshot of traces. In this case the Max RSS.
    :param key_type         : Group by the line number, defaults to 'lineno'.
    :param limit (int)      : Number of profiles to monitor, defaults to 20.
    """

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top {} lines".format(limit))
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        line = linecache.getline(frame.filename, frame.lineno).strip()
        print("#{:3d}: {:23s} | LineNo: {:>4} | RSS: {:>8} | LINE: {:>8}".format(index, filename, frame.lineno,
                                                                                 format_bytes(stat.size), line))
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("{} other calls: {}".format(len(other), format_bytes(size)))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: {}".format(format_bytes(total)))


def memory_monitor(command_queue: Queue, poll_interval=1):
    """Function to start the memory monitoring thread.
    Note:
        Starts tracemalloc trace and set max = 0 and snapshot = None then while True starts by removing and returning
        an item from the queue with a 0.1 second interval. This blocks queue at the most timeout seconds and
        raises the Empty from queue in the except if no item is available within that time. Then get_process_memory 
        is used to get the max_rss and tracemalloc.take_snapshot() starts tracing the block.
    :param command_queue            : Queue from the queue function.
    :param poll_interval (int)      : Set to 0.1 seconds, defaults to 1.
    """

    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now())
                display_top(snapshot)
            return
        except Empty:
            max_rss, _, _ = get_process_memory()
            if max_rss > old_max:
                old_max = max_rss
                snapshot = tracemalloc.take_snapshot()


def profile_memory(function, *args, **kwargs):
    """Profiles RSS Memory primarily and also prints our the VMS and Shared Memory.
    Note:
        get_process_memory is called to return RSS, VMS and Shared memory. Then the FIFO queue process is started
        and the poll_interval is set as 0.1 and passed in as arguments to memory_monitor inside Thread.
        Thread then spawns a separate thread for the specific memory instance and starts monitoring it. Note that
        Python doesn't actually do multi threading due to the GIL global interpreter locks which prevents multi 
        threading because the python memory management is thread safe. So there will be a little difference in how
        accurate the values are. Next start is called to start timing the function and the elapsed time is measured
        and the RSS, VMS and the Shared memory is measured and the queue is put to stop and the thread is joined.
    :param function         : Pass in the function to be memory profiled.
    :return                 : Memory Profile.
    """

    def wrapper(*args, **kwargs):
        rss_before, vms_before, shared_before = get_process_memory()
        queue = Queue()
        poll_interval = 0.1
        monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
        monitor_thread.start()
        start = time.time()
        result = function(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after, shared_after = get_process_memory()
        queue.put('stop')
        monitor_thread.join()
        print("Profiling: {:>20} RSS: {:>8} | VMS: {:>8} | SHR {:>8} | time: {:>8}".format(
            "<" + function.__name__ + ">",
            format_bytes(rss_after - rss_before),
            format_bytes(vms_after - vms_before),
            format_bytes(shared_after - shared_before),
            elapsed_time))
        return result

    print("Memory Profiling Complete!")
    if inspect.isfunction(function):
        return wrapper
    elif inspect.ismethod(function):
        return wrapper(*args, **kwargs)


def start_monitoring(args):
    if args.profile_type == 'time':
        profile_time(main)
    elif args.profile_type == 'memory':
        run_profiling = profile_memory(main)
        run_profiling()
