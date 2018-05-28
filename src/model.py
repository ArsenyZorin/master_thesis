import helpers
import tensorflow as tf
import numpy as np
from random import random

class Seq2seq:
    def __init__(self, encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights):
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.weights = weights

    def init_encoder(self):
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded,
            dtype=tf.float32, time_major=True,)

    def init_decoder(self):
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            self.decoder_cell, self.decoder_inputs_embedded,
            initial_state=self.encoder_final_state,
            dtype=tf.float32, time_major=True, scope="plain_decoder",)

        self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

    def init_optimizer(self):
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,)

        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, length, vocab, batches, directory):
        help_batch = helpers.random_sequences(
					length_from=length['from'], length_to=length['to'],
                    vocab_lower=vocab['lower'], vocab_upper=vocab['size'],
                    batch_size=batches['size'])

        saver = tf.train.Saver(self.seq2seq_vars)
        loss_track = []
        for batch in range(batches['max'] + 1):
            seq_batch = next(help_batch)
            fd = self.make_train_inputs(seq_batch, seq_batch)
            _, loss, state = self.sess.run([self.train_op, self.loss, self.encoder_final_state[0]], fd)
            loss_track.append(loss)
            print('\rBatch {}/{}\tloss: {}\tshape: {}'.format(batch, batches['max'], loss, state.shape), end="")

        print('\nLoss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1],
                         len(loss_track) * batches['size'],
                         batches['size']))
        path = saver.save(self.sess, directory + '/seq2seq.ckpt')
        print("Trained model saved to {}".format(path))

    def get_encoder_status(self, sequence, sec_seq=None):
        if type(sequence) is dict and type(sec_seq) is dict:
            first = {}
            second = {}
            for (k,v), (k2,v2) in zip(sequence.items(), sec_seq.items()):
                first[k] = self.get_stat(v)
                second[k2] = self.get_stat(v2)
                print('\rFirst:{}/{}\tSecond:{}/{}'.format(len(first), len(sequence),
                                    len(second), len(sec_seq)),end='')
            print()
            return first, second
        if type(sequence) is list or type(sequence) is np.ndarray:
            first = []
            second = []
            for line1, line2 in zip(sequence, sec_seq):
                first.append(self.get_stat([line1]))
                second.append(self.get_stat([line2]))
                print('\rFirst:{}/{}\tSecond:{}/{}'.format(len(first), len(sequence),
                                    len(second), len(sec_seq)), end='')
    def get_stat(self, elem):
        feed_dict = {self.encoder_inputs: np.transpose([elem])}
        stat = self.sess.run(self.encoder_final_state[0], feed_dict=feed_dict)
        return stat

class SiameseNetwork:
    def __init__(self, sequence_length, batch_size, layers):
        self.input_x1 = tf.placeholder(tf.float32, shape=(None, sequence_length), name='originInd')
        self.input_x2 = tf.placeholder(tf.float32, shape=(None, sequence_length), name='cloneInd')
        self.input_y = tf.placeholder(tf.float32, shape=None, name='answers')
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.layers = layers
        self.init_out()
        self.loss_accuracy_init()

    def init_out(self):
        self.out1 = self.rnn(self.input_x1, 'method1')
        self.out2 = self.rnn(self.input_x2, 'method2')
        self.distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.subtract(self.out1, self.out2))))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1))),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.out2)))))

    def loss_accuracy_init(self):
        self.temp_sim = tf.subtract(tf.ones_like(self.distance, dtype=tf.float32),
                                    self.distance, name='temp_sim')
        self.correct_predictions = tf.equal(self.temp_sim, self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'), name='accuracy')
        self.loss = self.get_loss()
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def get_loss(self):
        tmp1 = (1 - self.input_y) * tf.square(self.distance)
        tmp2 = self.input_y * tf.square(tf.maximum(0.0, 1 - self.distance))
        return tf.add(tmp1, tmp2) / 2

    def rnn(self, input_x, name):
        with tf.name_scope('fw' + name), tf.variable_scope('fw' + name):
            stacked_rnn_fw = []
            for _ in range(self.layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            stacked_rnn_bw = []
            for _ in range(self.layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, [input_x], dtype=tf.float32)
        return outputs

    def train(self, input_x1, input_x2, input_y, directory):
        batches = helpers.siam_batches(input_x1, input_x2, input_y)
        data_size = batches.shape[0]
        for nn in range(data_size):
            x1_batch, x2_batch = helpers.shape_diff(batches[nn][0], batches[nn][1])
            y_batch = batches[nn][2]
            feed_dict = self.dict_feed(x1_batch, x2_batch, y_batch)
            _, loss, dist, temp_sim = \
                self.sess.run([self.train_op, self.loss, self.distance, self.temp_sim], feed_dict)
            print('TRAIN: step {}/{}\tloss {:g} |\tExpected: {}\tGot: {}'.format(nn, data_size, loss, y_batch, dist))

        saver = tf.train.Saver(self.siam_vars)
        save_path = saver.save(self.sess, directory + '/siam.ckpt')
        print('Trained model saved to {}'.format(save_path))

    def eval_ds(self, dict, ds):
        clones = {}
        fp, tp, fn, tn = 0, 0, 0, 0
        for set in ds:
            if clones.get(set[0], 0) == 0:
                clones[set[0]] = []
            answ = self.step(dict[set[0]], dict[set[1]], set[2])
            tp += answ[0]
            fp += answ[1]
            fn += answ[2]
            tn += answ[3]
            if answ[0] == 1:
                clones[set[0]].append(set[1])
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        print('TP: {}\tRecall: {}\tPrecision: {}'.format(tp, recall, precision))

    def step(self, x1, x2, answ):
        tp, fp, fn, tn = 0, 0, 0, 0
        x1_batch, x2_batch = helpers.shape_diff(x1, x2)

        feed_dict = self.dict_feed(x1_batch, x2_batch)
        dist, sim = self.sess.run([self.distance, self.temp_sim], feed_dict)

        if int(answ) == 0 and int(round(dist)) == 0:
            tp = 1
        elif int(answ) == 0 and int(round(dist)) == 1:
            fn = 1
        elif int(answ) == 1 and int(round(dist)) == 1:
            tn = 1
        elif int(answ) == 1 and int(round(dist)) == 0:
            fp = 1

        return tp, fp, fn, tn
