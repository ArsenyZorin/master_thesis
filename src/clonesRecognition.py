import numpy as np
import tensorflow as tf
import json
import sys
import os

import time
import shutil
from model import Seq2seq, SiameseNetwork

tf.flags.DEFINE_string('type', 'full', 'Type of evaluation. Could be: \n\ttrain\n\teval\n\tfull')
tf.flags.DEFINE_string('data', os.path.expanduser('~/.rnncodeclones'), 'Directory with data for analysis')
tf.flags.DEFINE_integer('cpus', 1, 'Amount of threads for evaluation')
tf.flags.DEFINE_integer('gpus', None, 'Amount of GPUs for training')

FLAGS = tf.flags.FLAGS

def train(cell, layers, length, vocab, weights, batch, seq2seq_dir, siam_dir, vectors_dir):
    model = {'seq2seq': seq2seq_train(cell, length, vocab, weights, batch, seq2seq_dir)}
    model['siam'] = siam_train(vectors_dir, model['seq2seq'], batch['size'], layers, siam_dir)
    return model

def eval(model, vectors_dir):
    all,ds = readBcb(vectors_dir + '/test')
    states = model['seq2seq'].get_encoder_status(all)
    model['siam'].eval_ds(states, ds)

def seq2seq_train(cell, length, vocab, weights, batch, directory):
    seq2seq_model = Seq2seq(cell['encoder'], cell['decoder'], vocab['size'], weights.shape[1], weights)
    seq2seq_model.train(length, vocab, batch, directory)
    return seq2seq_model

def siam_train(vectors, seq2seq_model, batch_size, layers, directory):
    first, sec, answ = readBcb(vectors + '/train')

    first_enc = seq2seq_model.get_encoder_status(first)
    sec_enc = seq2seq_model.get_encoder_status(sec)

    siam_model = SiameseNetwork(first_enc[0].shape[1], batch_size, layers)
    siam_model.train(first_enc, sec_enc, answ, directory)
    return siam_model

def main(_):
    start = time.time()
    try:
        tf.reset_default_graph()
        print(tf.__version__)

        if FLAGS.type != 'eval' and FLAGS.type != 'train' and FLAGS.type != 'full':
            print('Unknown type flag.')
            print('Allowable values are:')
            print('\ttrain\n\teval\n\tfull')
            show_time(start)
            sys.exit(1)

        dirs = {'seq2seq': FLAGS.data + '/networks/seq2seqModel',
               'siam': FLAGS.data + '/networks/siameseModel',
               'vecs': FLAGS.data + '/vectors'}

        if FLAGS.type != 'eval':
            if os.path.exists(dirs['seq2seq']):
                shutil.rmtree(dirs['seq2seq'])
            if os.path.exists(dirs['siam']):
                shutil.rmtree(dirs['siam'])
            os.mkdir(dirs['seq2seq'])
            os.mkdir(dirs['siam'])

        weights_file = open(FLAGS.data + '/networks/word2vec/pretrainedWeights', 'r')
        weights = np.array(json.loads(weights_file.read()))

        vocab = {'size': weights.shape[0], 'lower': 2}
        length = {'from': 1, 'to': 100}
        batch = {'size': 1000, 'max': 1500, 'epoch': 1000}

        layers = 10
        encoder_hidden_units = 10
        decoder_hidden_units = encoder_hidden_units

        cell = {'encoder': tf.contrib.rnn.LSTMCell(encoder_hidden_units),
                'decoder': tf.contrib.rnn.LSTMCell(decoder_hidden_units)}

        if FLAGS.type == 'train':
            train(cell, layers, length, vocab, weights, batch, dirs['seq2seq'], dirs['siam'], dirs['vecs'])
        elif FLAGS.type == 'full':
            model = train(cell, layers, length, vocab, weights, batch, dirs['seq2seq'], dirs['siam'], dirs['vecs'])
            eval(model, dirs['vecs'])
        elif FLAGS.type == 'eval':
            model = restore_models(dirs, cell, vocab, length, weights, batch, layers)
            eval(model, dirs['vecs'])

        show_time(start)

    except KeyboardInterrupt:
        print('Keyboard interruption')
        show_time(start)
        sys.exit(0)

if __name__ == '__main__':
    tf.app.run()
