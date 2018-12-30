import numpy as np
import tensorflow as tf
import torch


class Logger(object):
    """
    Arguments:

    Returns:

    Note: Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

    """

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        :param tag:
        :param value:
        :param step:
        :return:
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """
        :param tag:
        :param values:
        :param step:
        :param bins:
        :return:
        """
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def save_models(model, path, epoch, optimizer, best, loss):
    """
    :param model:
    :param path:
    :param epoch:
    :param optimizer:
    :param best:
    :param loss:
    :return:
    """
    if best:
        print("===> Saving a new best model at epoch {}".format(epoch))
        save_checkpoint = ({'model': model,
                            'optimizer': optimizer,
                            'epoch': epoch,
                            'best_loss': loss
                            }, best)
        torch.save(save_checkpoint, path + "/u_net_model.pt")
