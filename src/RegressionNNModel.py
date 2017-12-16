"""
Neural Regression Model
"""
from LaIRNNModel import LaIRNNModel
import tensorflow as tf

class RegressionNNModel(LaIRNNModel):
    """
    Implements a feedforward neural network
    This network will predict the exact {wait, help, total} time for a LaIR request (in minutes).
    """
    def labelType(self):
        return tf.float32

    def add_output_op(self, net):
        pred = tf.contrib.layers.fully_connected(net, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization))
        return pred

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
        return tf.losses.mean_squared_error(self.labels_placeholder, pred) + tf.reduce_sum(reg_losses)

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
        return sess.run(self.loss, feed_dict=feed)

    def getLabelFn(self, loader, timeChar):
        """Returns a function that maps from a single training example to its label"""
        return self.getLabelLambda(timeChar)