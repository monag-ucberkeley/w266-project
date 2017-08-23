import json, os, re, shutil, sys, time, csv
import collections, itertools
from collections import Counter
import operator
import os, sys, re, json, time, csv, copy
import time
import itertools, collections
import pandas as pd
import numpy as np
import glob

# NLTK for NLP utils and corpora
import nltk
assert(nltk.download('gutenberg'))

# NumPy and TensorFlow
import numpy as np
import tensorflow as tf
assert(tf.__version__.startswith("1."))

# utils.pretty_print_matrix uses Pandas. Configure float format here.
import pandas as pd
pd.set_option('float_format', lambda f: "{0:.04f}".format(f))

import utils, vocabulary, rnnlm

def run_epoch(lm, session, batch_iterator,
              train=False, verbose=False,
              tick_s=10, learning_rate=0.01):
    start_time = time.time()
    tick_time = start_time  # for showing status
    total_cost = 0.0  # total cost, summed over all words
    total_batches = 0
    total_words = 0

    if train:
        train_op = lm.train_step_
        use_dropout = True
        loss = lm.train_loss_
    else:
        train_op = tf.no_op()
        use_dropout = False  # no dropout at test time
        loss = lm.loss_  # true loss, if train_loss is an approximation

    for i, (w, y) in enumerate(batch_iterator):
        cost = 0.0
        
        # At first batch in epoch, get a clean intitial state.
        if i == 0:
            h = session.run(lm.initial_h_, {lm.input_w_: w})

        #### YOUR CODE HERE ####
        feed_dict = {lm.input_w_:w,
                     lm.target_y_:y,
                     lm.initial_h_:h,
                     lm.dropout_keep_prob_: 1.0}
        
        if train:
            cost, train_op_ = session.run([lm.train_loss_, lm.train_step_], feed_dict)
        else:
            cost, train_op_ = session.run([lm.loss_, lm.train_step_], feed_dict)

        ## print "Calling computational graph"
        #cost, train_op_ = session.run([loss, train_op], feed_dict)
        #### END(YOUR CODE) ####
        total_cost += cost
        total_batches = i + 1
        total_words += w.size  # w.size = batch_size * max_time
        
        ##
        # Print average loss-so-far for epoch
        # If using train_loss_, this may be an underestimate.
        if verbose and (time.time() - tick_time >= tick_s):
            avg_cost = total_cost / total_batches
            avg_wps = total_words / (time.time() - start_time)
            print "[batch %d]: seen %d words at %d wps, loss = %.3f" % (
                i, total_words, avg_wps, avg_cost)
            tick_time = time.time()  # reset time ticker
            
        if avg_cost <= 0.00001:
            break

    return total_cost / total_batches

def score_dataset(lm, session, ids, name="Data"):
    # For scoring, we can use larger batches to speed things up.
    bi = utils.batch_generator(ids, batch_size=100, max_time=100)
    cost = run_epoch(lm, session, bi, 
                     learning_rate=1.0, train=False, 
                     verbose=False, tick_s=3600)
    print "%s: avg. loss: %.03f  (perplexity: %.02f)" % (name, cost, np.exp(cost))


def get_character_tokens_and_sents():
         
	dialogsByCharacter = {}
	indir = '../../../book-nlp-master/data/tokens.gutenberg/'
        print "PATH -> ", glob.glob(os.path.join("../../../book-nlp-master/data/tokens.gutenberg/", "Sir_Arthur_Conan_Doyle*.tokens"))
        #Read all works by Sir Arthur Conan Doyle
	df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(indir, "Sir_Arthur_Conan_Doyle*.tokens"))))
	#df = pd.read_csv(indir + '/Sir_Arthur_Conan_Doyle___The_Hound_of_the_Baskervilles.tokens', sep='\t', quoting=csv.QUOTE_NONE)
	#df.query('characterId > -1 & inQuotation == False')
	#for i in xrange(df['characterId'].max()):
    	sentenceIDs = []
    	sentenceIDs.append(df.query('characterId == 9 & (pos == "PRP$" | pos == "PRP")')['sentenceID'].unique())
    	#for sentenceID in sentenceIDs:
    	#print sentenceIDs
    	#if sentenceIDs.size:
        #	print "Character - ", i
        for sentenceID in np.nditer(sentenceIDs):
		#print sentenceID
            	dialogs = []
            	#df.query('sentenceID == ' + str(sentenceID) + ' & inQuotation == True')['originalWord']
            	dialogs.append(df.query('sentenceID == ' + str(sentenceID) + ' & inQuotation == True')['originalWord'])
        #dialogsByCharacter[i] = dialogs
	print dialogs



def loadAndPreprocessData():

	'''
	Read all the words to create a vocabulary
	'''
	all_tokens = []
	indir = '../preprocess/'

	for root, dirs, filenames in os.walk(indir):
	    for filename in filenames:
	        if filename.startswith('canonicalized_words_'):
	            with open(indir+filename, 'r') as f:
	                for line in f.readlines():
	                    w = line.rstrip()
	                    if w != '':
	                        all_tokens.append(w)
	print 'Processed all tokens: ', len(all_tokens)

	tokens_dict = Counter()
	for w in all_tokens:
	    if w.startswith('DG') and w.endswith('DG'):
	        w = 'DG'
	    tokens_dict[w] += 1

	'''
	Remove noisy tokens - see notebook for exploratory analysis
	The first ~2500 tokens when sorted by key are noisy like "!!!!" or "* * * *" - for eg, the end of a chapter
	'''
	noisy_tokens = sorted(tokens_dict)[0:2507]
	print 'Identified noisy tokens - some examples: ', noisy_tokens[0:30]
	'''
	Clean up the tokens now that we know the noisy tokens and then generate the vocab
	'''
	noisy_tokens = set(noisy_tokens)
	words = [ w for w  in all_tokens if w not in noisy_tokens ]
	# TODO: Should make V configurable
	V = 50000
	vocab = vocabulary.Vocabulary((word for word in words), size=V)
	print 'Vocabulary created with size: ', vocab.size

	'''
	Read in the sentences already parsed from the ~3000 books Gutenberg subset
	'''
	sents = []
	indir = '../preprocess/'
	books = []
	for root, dirs, filenames in os.walk(indir):
	    for filename in filenames:
	        if filename.startswith('parsed_sents_'):
	            with open(indir+filename, 'r') as f:
	                for line in f.readlines():
	                    sents.append(line.rstrip())
	print 'Parsed sentences loaded into memory: ', len(sents)
	print 'The 10,000th sentence is: ', sents[10000]
	'''
	Prepare training and test sentences
	'''
	split=0.8 
	shuffle=True

	sentences = np.array(sents, dtype=object)
	fmt = (len(sentences), sum(map(len, sentences)))
	print "Loaded %d sentences (%g tokens)" % fmt

	if shuffle:
	    rng = np.random.RandomState(shuffle)
	    rng.shuffle(sentences)  # in-place
	train_frac = 0.8
	split_idx = int(train_frac * len(sentences))
	train_sentences = sentences[:split_idx]
	test_sentences = sentences[split_idx:]

	fmt = (len(train_sentences), sum(map(len, train_sentences)))
	print "Training set: %d sentences (%d tokens)" % fmt
	fmt = (len(test_sentences), sum(map(len, test_sentences)))
	print "Test set: %d sentences (%d tokens)" % fmt

	'''
	Apply the vocab to the train and test sentences and convert words to ids to start training
	'''
	## Preprocess sentences
	## convert words to ids based on the vocab wordset created above
	## Do this in batches to avoid crashes due to insufficient memory
	batch_size = 50000
	num_of_batches = int(round(len(train_sentences) / batch_size))
	print 'Preprocessing train sentences - number of batches: ', num_of_batches
	train_id_batches = []
	start = 0
	end = start + batch_size
	for i in range(num_of_batches): 
	    if i % 15 is 0:
	    	print 'Completed Batches: ', i
	    train_id_batches.append(utils.preprocess_sentences(train_sentences[start:end], vocab))
	    start = end
	    end += batch_size
	# flatten the lists for 1D tensor
	temp = utils.flatten(train_id_batches)
	train_ids = utils.flatten(temp)
	print 'Train sentences converted to their IDs including start, end token and unknown word token'

	# repeat the same with test data
	batch_size = 50000
	num_of_batches = int(round(len(test_sentences) / batch_size))
	if num_of_batches > 10:
	    num_of_batches = 10
	print 'Preprocessing test sentences - number of batches: ', num_of_batches
	test_id_batches = []
	start = 0
	end = start + batch_size
	for i in range(num_of_batches): 
	    print 'Batch: ', i
	    test_id_batches.append(utils.preprocess_sentences(test_sentences[start:end], vocab))
	    start = end
	    end += batch_size
	test_ids = utils.flatten(utils.flatten(test_id_batches))

	print 'Test sentences converted to their IDs including start, end token and unknown word token'
	max_time = 40
	batch_size = 64
	learning_rate = 0.01
	num_epochs = 3

	# Model parameters
	model_params = dict(V=vocab.size, 
	                    H=100, 
	                    softmax_ns=200,
	                    num_layers=1)

	TF_SAVEDIR = "tf_saved"
	checkpoint_filename = os.path.join(TF_SAVEDIR, "rnnlm")
	trained_filename = os.path.join(TF_SAVEDIR, "rnnlm_trained")
	# Will print status every this many seconds
	print_interval = 120

	# Clear old log directory
	shutil.rmtree("tf_summaries", ignore_errors=True)

	lm = rnnlm.RNNLM(**model_params)
	lm.BuildCoreGraph()
	lm.BuildTrainGraph()

	# Explicitly add global initializer and variable saver to LM graph
	with lm.graph.as_default():
	    initializer = tf.global_variables_initializer()
	    saver = tf.train.Saver()
	    
	# Clear old log directory
	shutil.rmtree(TF_SAVEDIR, ignore_errors=True)
	if not os.path.isdir(TF_SAVEDIR):
	    os.makedirs(TF_SAVEDIR)

	with tf.Session(graph=lm.graph) as session:
	    # Seed RNG for repeatability
	    tf.set_random_seed(42)
	    session.run(initializer)

	    for epoch in xrange(1,num_epochs+1):
	        t0_epoch = time.time()
	        bi = utils.batch_generator(train_ids, batch_size, max_time)
	        print "[epoch %d] Starting epoch %d" % (epoch, epoch)
	        #### YOUR CODE HERE ####
	        # Run a training epoch.
	        
	        run_epoch(lm, session, bi, train=True, learning_rate=learning_rate)
	        
	        #### END(YOUR CODE) ####
	        print "[epoch %d] Completed in %s" % (epoch, utils.pretty_timedelta(since=t0_epoch))
	    
	        # Save a checkpoint
	        saver.save(session, checkpoint_filename, global_step=epoch)
		    ##
		    # score_dataset will run a forward pass over the entire dataset
		    # and report perplexity scores. This can be slow (around 1/2 to 
		    # 1/4 as long as a full epoch), so you may want to comment it out
		    # to speed up training on a slow machine. Be sure to run it at the 
		    # end to evaluate your score.
		print ("[epoch %d]" % epoch), score_dataset(lm, session, train_ids, name="Train set")
		print ("[epoch %d]" % epoch), score_dataset(lm, session, test_ids, name="Test set")
		print ""
	    # Save final model
	    saver.save(session, trained_filename)

if __name__ == '__main__':
    #loadAndPreprocessData()
    get_character_tokens_and_sents()


