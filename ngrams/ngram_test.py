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

#what is the length of the model we should use
GRAM_LENGTH = 3


CORPUS_PATH = "corpus.txt"
def get_corpus():
    with open(CORPUS_PATH) as corpus_text:
        return corpus_text.read().replace('\n', ' ').lower()


corpus = get_corpus()


token = nltk.word_tokenize(corpus)
ngrams = ngrams(token, GRAM_LENGTH)

#counts the number of times a word follows a sequence of words in the corpus
ngram_dict = defaultdict(Counter)

for ngram in ngrams:
    lead = tuple(ngram[:-1]) #leading n-1gram
    follow = ngram[-1] #resultant word

    ngram_dict[lead][follow] += 1


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
prev_words = tuple(["and", "you"])

#keeps track of the sentence we have generated
generated = ''
for word in prev_words:
    generated += word
    generated += ' '


#dummy value
generated_word = tuple(' ')

while generated_word[0] != ".":
    
    #must get counts of each following word to pass to random.choices
    words = list()
    probs = list()
    for follow, prob in ngram_dict[prev_words].items():
        words.append(follow)
        probs.append(prob)

    #choose the new word and put it in a tuple
    generated_word = tuple(choices(words, k=1, weights=probs))

    #add to generated sentence
    generated += generated_word[0]
    generated += ' '

    #remove oldest leading word and add newly generated word
    prev_words = prev_words[1:] + generated_word

print(generated)
