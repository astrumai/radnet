"""
Script for multi gpu training and multi processing/ no multi threading because I am writing in python

to check if cuda is available:
    torch.cuda.is_available()

to check cuda name:
    torch.cuda.get_device_name(0)

"""


class MultiGpu(object):

    def __init__(self, workers):
        self.workers = workers

    def get_num_devices(self):
        """Get how many CUDA devices are present"""
        pass

    def mem_info(self):
        """Get memory info about the CUDA devices"""
        pass

    def devices(self):
        """Return those devices to be used for training"""
        pass


class MultiProcessing(object):

    def __init__(self, *args, **kwargs):
        pass
