import logging

# I can also see if I can add a log file that saves in the hyper parameter I use

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('test.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


from functools import wraps


def my_logger(original_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(original_func.__name__), level=logging.INFO)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        logging.INFO(
            'Ran with args:{}, and kwargs: {}'.format(args, kwargs)
        )
        return original_func
    return wrapper


# I can write a similar timing function and wrap it using the decorator to log the time
def my_timer(original_func):
    import time

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = original_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(original_func.__name__, t2))
        return result
    return wrapper


# the following decorator is equal to
# display = my_logger(my_timer(display_info))
@my_logger
@my_timer
def display_info(name, age):
    print('display_info ran with arguments ({}, {})'.format(name, age))
