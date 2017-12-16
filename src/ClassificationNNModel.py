"""
Neural Classification Model
"""
from util import make_buckets, make_bucket_mapper, ConfusionMatrix
from LaIRNNModel import LaIRNNModel
import tensorflow as tf
import numpy as np

class ClassificationNNModel(LaIRNNModel):
    """
    Implements a feedforward neural network
    This network will predict the bucketed {wait, help, total} time for a LaIR request (in minutes).
    """

    def labelType(self):
        return tf.int32

    def add_output_op(self, net):
        """Adds the output layer:
            pred = net . U + b

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Here are the dimensions of the various variables:
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        U_shape = (self.config.hidden_size, int(self.config.bucketString[:-1]))
        b_shape = (int(self.config.bucketString[:-1]), )
        U = tf.get_variable('U', shape=U_shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b2', shape=b_shape)
        return tf.matmul(net, U) + b

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using MSE loss.
        The loss should be averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size,) containing the output of the neural network
        Returns:
            loss: A 0-d tensor (scalar)
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder, name='softmax')
        loss = tf.reduce_mean(losses) + tf.reduce_sum(reg_losses)
        return loss

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, log=True):
        """Return the loss after evaluating on the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples,)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        output_pred = tf.argmax(tf.nn.softmax(self.pred), axis=1)
        output = sess.run(output_pred, feed_dict=feed)

        num_correct = 0
        if log: confusion_matrix = ConfusionMatrix(np.sort(np.unique(labels_batch)))
        for i in range(inputs_batch.shape[0]):
            y = labels_batch[i]
            y_hat = output[i]
            if log: confusion_matrix.update(y, y_hat)
            if y == y_hat:
                num_correct += 1
            # else:
                # print("pred was {}, truth was {}".format(y_hat, y))

        if log: print(confusion_matrix.as_table())  

        return 1 - (1.0 * num_correct / inputs_batch.shape[0])

    def getLabelFn(self, loader, timeChar):
        """Returns a function that maps from a single training example to its label"""

        buckets = make_buckets(loader, self.config.bucketString, timeChar)
        return self.getLabelLambda(timeChar, bucket_mapper=make_bucket_mapper(buckets))

