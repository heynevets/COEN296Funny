"""
First attempt at sentence generator using ngram model

Appears to work as expected, generates a single sentence, stopping when
    a '.' is randomly generated

Todo:
    Needs a fallback if previous words were not found 
        (fall back to bigram or unigram model)
"""

import nltk
from nltk.util import ngrams
from collections import Counter, defaultdict
from random import choices
import pickle

#what is the length of the model we should use
GRAM_LENGTH = 3
LOAD_DICT = True
END = tuple(["sent_e2", "sent_e1"])
START = tuple(["sent_s1", "sent_s2"])


CORPUS_PATH = "shortjokes.csv"
def get_corpus():
    with open(CORPUS_PATH) as corpus_text:
        return corpus_text.read().replace('\n', ' ').lower()



#counts the number of times a word follows a sequence of words in the corpus

forward_dicts = list()
backward_dicts = list()

if LOAD_DICT:
    '''
    for i in range(GRAM_LENGTH, 0, -1):
        forward_dicts.append(pickle.load(open(str(i)+"gram_model.pickle", 'rb')))
        backward_dicts.append(pickle.load(open(str(i)+"gram_model_reversed.pickle", 'rb')))
    '''
    forward_dicts.append(pickle.load(open("3gram_model.pickle", 'rb')))
    
else:
    corpus = get_corpus()


    token = list(nltk.word_tokenize(corpus))

    for i in range(GRAM_LENGTH, 0, -1):
        print(i)
        igrams = list(ngrams(token, i))

        skip_punc = set(["'", '"', "(", ")", "''", '""', '...', '[', ']', '`', '``'])
        ngram_dict = defaultdict(Counter)
        for ngram in igrams:
            lead = tuple(ngram[:-1]) #leading n-1gram
            follow = ngram[-1] #resultant word
            if follow in skip_punc or follow in START:
                continue

            ngram_dict[lead][follow] += 1

        pickle.dump(ngram_dict, open(str(i) + "gram_model.pickle", 'wb'))
        forward_dicts.append(ngram_dict)

        #build reverse dict
        reversed_ngram_dict = defaultdict(Counter)
        for ngram in igrams:
            ngram = list(reversed(ngram))
            lead = tuple(ngram[:-1]) #leading n-1gram
            follow = ngram[-1] #resultant word
            if follow in skip_punc:
                continue

            ngram_dict[lead][follow] += 1

        pickle.dump(ngram_dict, open(str(i)+ "gram_model_reversed.pickle", 'wb'))
        backward_dicts.append(ngram_dict)


#function to calculate probabilities, unnecessary because random.choices allows
#us to pass in raw counts and it creates a distribution from them
'''
prob_dict = defaultdict(dict)
for lead, counter in ngram_dict.items():
    total_count = len(counter)
    for follow, count in counter.items():
        prob_dict[lead][follow] = count / total_count

print(prob_dict)
'''

#seed words
print("Ready!")

#keeps track of the sentence we have generated
while True:
    prev_words = tuple(input().lower().split())

    generated = ''
    for word in prev_words:
        generated += word
        generated += ' '


    #dummy value
    generated_word = tuple(' ')

    stop_punc = set(['!', '.'])
    while prev_words[-1] not in stop_punc:
        #must get counts of each following word to pass to random.choices
        backoff = GRAM_LENGTH - len(prev_words)
        words = list()
        probs = list()
        '''
        while not words:
            p_words = prev_words[backoff:]
        '''

        for follow, prob in forward_dicts[0][prev_words].items():
            words.append(follow)
            probs.append(prob)

        #backoff += 1

        #choose the new word and put it in a tuple
        try:
            generated_word = tuple(choices(words, k=1, weights=probs))
        except IndexError:
            break

        #add to generated sentence
        if generated_word[0] not in START and generated_word[0] not in END:
            generated += generated_word[0]
            generated += ' '

        #remove oldest leading word and add newly generated word
        prev_words = prev_words + generated_word
        if len(prev_words) > GRAM_LENGTH-1:
            prev_words = prev_words[1:]

    print(generated)
