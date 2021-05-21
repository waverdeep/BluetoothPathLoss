from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def set_tensorboard_writer(name):
    writer = SummaryWriter(name)
    return writer


def close_tensorboard_writer(writer):
    writer.close()
