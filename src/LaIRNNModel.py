"""
(Adapted from CS224N Winter 2017 Assignment 3)

Neural LaIR Model
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import time
from datetime import datetime
import tensorflow as tf
from data.DataLoader import DataLoader
from util import get_minibatches

class LaIRNNModel:
    """
    Implements a feedforward neural network
    This network will predict the {wait, help, total} time for a LaIR request (in minutes).
    """

    def __init__(self, config):
        self.config = config
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None,), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_features), name='input')
        self.labels_placeholder = tf.placeholder(self.labelType(), shape=(None), name='labels')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout')

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        The keys for the feed_dict should be a subset of the placeholder
        tensors created in add_placeholders.
        
        When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        """Adds the N-hidden-layer NN:
            h1 = Elu(BatchNorm(xW1 + b1))
            h1_drop = Dropout(h1, dropout_rate)
            ...
            hn_drop = Dropout(hn, dropout_rate)
            pred = output_op(hn_drop)

        Each layer has a regularization operation for the weights.

        Note: tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of dropout_rate.

        Returns:
            pred: tf.Tensor of shape (batch_size,)
        """

        curr = self.input_placeholder
        dropout_rate = self.dropout_placeholder

        # Make n_hidden_layers fully connected layers
        for i in range(self.config.n_hidden_layers):
            curr = tf.contrib.layers.fully_connected(curr, self.config.hidden_size,
                activation_fn=tf.nn.elu,
                normalizer_fn=tf.contrib.layers.batch_norm,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization))
            curr = tf.contrib.layers.dropout(curr, keep_prob=dropout_rate)

        return self.add_output_op(curr)

    def add_output_op(self, net):
        """Adds an op for the output layer, given the output of the last hidden layer"""
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        The loss should be averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size,) containing the output of the neural network
        Returns:
            loss: A 0-d tensor (scalar)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor
        Returns:
            train_op: The Op for training.
        """
        return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

    def run_epoch(self, sess, train_writer, inputs, labels, epochNum):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            train_writer: writer object to write to TensorBoard logs
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples,)
            epochNum: the number (0-indexed) of the epoch we should run
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """

        # Get all of the Tensorboard summary nodes
        merged = tf.summary.merge_all()

        n_batches = int(inputs.shape[0] / self.config.batch_size) + (1 if inputs.shape[0] % self.config.batch_size > 0 else 0)

        total_loss = 0
        for i, (input_batch, labels_batch) in enumerate(get_minibatches([inputs, labels], self.config.batch_size, shuffle=False)):
            loss, output_summary = self.train_on_batch(sess, merged, input_batch, labels_batch)
            total_loss += loss

            # Add the summaries to our TensorBoard output
            train_writer.add_summary(output_summary, epochNum * n_batches + i)

        return 1.0 * total_loss / n_batches

    def fit(self, sess, saver, inputs, labels, log=True):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            saver: tf.Saver object to write model parameters to disk
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples,)
            log=True: whether to print log messages
        Returns:
            losses: list of loss per epoch
        """
        # Tensorboard logging
        train_writer = tf.summary.FileWriter(self.config.output_path + "logs/", tf.get_default_graph())

        # Track all avg losses for each epoch
        lowest_loss = float("inf")
        losses = []

        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, train_writer, inputs, labels, epoch)
            duration = time.time() - start_time
            if log: print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)

            # Update our lowest loss if needed
            if average_loss < lowest_loss:
                lowest_loss = average_loss
                if log: print("New best score! Saving model in %s" % self.config.model_output)
                saver.save(sess, self.config.model_output)
            if log: print("")

        train_writer.close()
        return losses

    def run(self, loader, vectorizers, timeChar, run_type="train", log=True):
        """Runs the model on the given time dataset, either in train mode or test mode

        Args:
            loader: The DataLoader to use to access data
            timeChar: either 'w' or 'h' for wait/help
            inputs: np.ndarray of shape (n_samples, n_features)
            run_type="train": either "train", "dev", "test"
            log=True: whether to print log messages
        Returns:
            nothing if training, evaluation loss otherwise
        """

        # Generate the x and y sets
        labels = loader.getLabels(self.getLabelFn(loader, timeChar))
        inputs = loader.applyVectorizers(vectorizers, run_type, timeChar, log=log)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            if run_type == "train":
                session.run(init)
                self.fit(session, saver, inputs, labels, log=log)
            else:
                saver.restore(session, self.config.model_output)
                return self.evaluate_on_batch(session, inputs, labels, log=log)

    def train_on_batch(self, sess, merged_summary, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            merged_summary: the summary node that outputs the loss to TensorBoard
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples,)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, 
            dropout=self.config.dropout)
        _, output_summary, loss = sess.run([self.train_op, merged_summary, self.loss], feed_dict=feed)
        return loss, output_summary

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, log=True):
        """Return the loss after evaluating on the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples,)
            log=True: whether to print log messages
        Returns:
            loss: loss over the batch (a scalar)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples,)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        """Builds up the tensorflow graph"""
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

        # Add a loss output to the graph for TensorBoard
        tf.summary.scalar('loss', self.loss)

    # Returns a function that, given a lair request, will return its
    # wait time if ch == 'w', help time if ch == 'h', and total time if ch == 't
    # if bucket_mapper is provided, calls bucket_mapper on that time before returning
    def getLabelLambda(self, ch, bucket_mapper=lambda x: x):
        def labelFn(x):
            if ch == 'w':
                return bucket_mapper(x.getWaitTimeMinutes())
            elif ch == 'h':
                return bucket_mapper(x.getHelpTimeMinutes())
            elif ch == 't':
                return bucket_mapper(x.getTotalTimeMinutes())
            else:
                return 0

        return labelFn

    def getLabelFn(self, loader, timeChar):
        """Returns a function that maps from a single training example to its label"""
        raise NotImplementedError("Each Model must re-implement this method.")

