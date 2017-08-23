import numpy as np
import tensorflow as tf
import nltk
import json, os, re, shutil, sys, time, csv
import collections, itertools
from collections import Counter
import operator
import pandas as pd
pd.set_option('float_format', lambda f: "{0:.04f}".format(f))
import utils, vocabulary, rnnlm


def sample_step(lm, session, input_w, initial_h):
    """Run a single RNN step and return sampled predictions.
  
    Args:
      lm : rnnlm.RNNLM
      session: tf.Session
      input_w : [batch_size] vector of indices
      initial_h : [batch_size, hidden_dims] initial state
    
    Returns:
      final_h : final hidden state, compatible with initial_h
      samples : [batch_size, 1] vector of indices
    """
    # Reshape input to column vector
    input_w = np.array(input_w, dtype=np.int32).reshape([-1,1])
  
    #### YOUR CODE HERE ####
    # Run sample ops
    final_h, samples = session.run([lm.final_h_, lm.pred_samples_], 
    feed_dict={lm.input_w_: input_w, lm.initial_h_: initial_h, lm.dropout_keep_prob_:1.0, lm.learning_rate_:0.1})


    #### END(YOUR CODE) ####
    # Note indexing here: 
    #   [batch_size, max_time, 1] -> [batch_size, 1]
    return final_h, samples[:,-1,:]

# Same as above, but as a batch
max_steps = 20
num_samples = 10
random_seed = 42

lm = rnnlm.RNNLM(**model_params)
lm.BuildCoreGraph()
lm.BuildSamplerGraph()

with lm.graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=lm.graph) as session:
    # Seed RNG for repeatability
    tf.set_random_seed(random_seed)
    
    # Load the trained model
    saver.restore(session, trained_filename)

    # Make initial state for a batch with batch_size = num_samples
    w = np.repeat([[vocab.START_ID]], num_samples, axis=0)
    h = session.run(lm.initial_h_, {lm.input_w_: w})
    # We'll take one step for each sequence on each iteration 
    for i in xrange(max_steps):
        h, y = sample_step(lm, session, w[:,-1:], h)
        w = np.hstack((w,y))

    # Print generated sentences
    for row in w:
        for i, word_id in enumerate(row):
            print vocab.id_to_word[word_id],
            if (i != 0) and (word_id == vocab.START_ID):
                break
        print ""
