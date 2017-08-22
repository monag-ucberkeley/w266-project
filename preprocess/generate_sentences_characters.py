import os, sys, re, json, time, csv, copy
import time
import itertools, collections
import pandas as pd

# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]


start_index = int(sys.argv[1])
print 'Processing from book: ', start_index
batch_size = 200

indir = '../book-nlp-master/data/tokens.gutenberg'
books = []
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
         books.append(f)
print 'Total number of books to process: ', batch_size
books = sorted(books)
books = books[start_index:start_index + batch_size]

sents = []
canonicalized_words = []
filename = "parsed_sents_" + str(start_index) + ".txt"
f = open(filename, "a")

for book_num, book in enumerate(books):    
    print "[ {0} ] Processing book: {1}".format(book_num, book)
    df = pd.read_csv(indir + '/' + book, sep='\t', quoting=csv.QUOTE_NONE)
    for i in xrange(df["sentenceID"].max()):
        tokens = []
        add_sentence = False
        for word in df.query('sentenceID == '+ str(i))['originalWord']:
            add_sentence = True
            if type(word) != str:
                word = str(word)
            word = word.decode('ascii', 'ignore')
            tokens.append(word)
            canonicalized_words.append(canonicalize_word(word))
        if add_sentence:
            sent = "<s> "
            sent += " ".join(tokens)
            sent += " </s>\n"
            #print sent
            sents.append(sent)
            f.write(sent)

f.close()
with open('canonicalized_words_' + str(start_index) + '.txt', 'a') as c:
    for word in canonicalized_words:
        c.write(word + "\n")

