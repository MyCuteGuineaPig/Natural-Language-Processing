#### Word Embedding 


**Data preparation**


```python
import re
import nltk
import emoji
import numpy as np
from nltk.tokenize import word_tokenize

corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'
data = re.sub(r'[,!?;-]+', '.', corpus) # replace all interrupting punctuation signs — such as commas and exclamation marks — with periods.
# data: Who ❤️ "word embeddings" in 2020. I do.

data = nltk.word_tokenize(data)
# data : ['Who', '❤️', '``', 'word', 'embeddings', "''", 'in', '2020', '.', 'I', 'do', '.']

# Filter tokenized corpus using list comprehension
data = [ ch.lower() for ch in data
         if ch.isalpha()
         or ch == '.'
         or emoji.get_emoji_regexp().search(ch)
       ]
#After cleaning:  ['who', '❤️', 'word', 'embeddings', 'in', '.', 'i', 'do', '.']

#-----------------------------------------------Generate One Hot Vector for Center/Conext-----------------------------------------------------

def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector

word2Ind  = {'am': 0, 'because': 1, 'happy': 2, 'i': 3, 'learning': 4}
Ind2word = {0: 'am', 1: 'because', 2: 'happy', 3: 'i', 4: 'learning'}

word_to_one_hot_vector('happy', word2Ind, V)
#array([0., 0., 1., 0., 0.])


context_words = ['i', 'am', 'because', 'i']
context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
#[array([0., 0., 0., 1., 0.]),
# array([1., 0., 0., 0., 0.]),
# array([0., 1., 0., 0., 0.]),
# array([0., 0., 0., 1., 0.])]

context_words_vectors = np.mean(context_words_vectors, axis=0)
# array([0.25, 0.25, 0.  , 0.5 , 0.  ])

#-----------------------------------------------Build training set-----------------------------------------------------

def get_windows(words, C):
    #C, is the context half-size. Recall that for a given center word, the context words are made of C words to the left and C words to the right of the center word.
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

words = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning']
for context_words, center_word in get_windows(words, 2):  # reminder: 2 is the context half-size
    print(f'Context words:  {context_words} -> {context_words_to_vector(context_words, word2Ind, V)}')
    print(f'Center word:  {center_word} -> {word_to_one_hot_vector(center_word, word2Ind, V)}')
    print()
"""
Context words:  ['i', 'am', 'because', 'i'] -> [0.25 0.25 0.   0.5  0.  ]
Center word:  happy -> [0. 0. 1. 0. 0.]

Context words:  ['am', 'happy', 'i', 'am'] -> [0.5  0.   0.25 0.25 0.  ]
Center word:  because -> [0. 1. 0. 0. 0.]

Context words:  ['happy', 'because', 'am', 'learning'] -> [0.25 0.25 0.25 0.   0.25]
Center word:  i -> [0. 0. 0. 1. 0.]
"""

```
**Activation functions**

```python
#-----------------------------------------------ReLU-----------------------------------------------------

def relu(z):
    result = z.copy()
    result[result < 0] = 0
    return result

z = np.array([[-1], [ 4], [ 2], [ 1], [-3]])
#  array([[0],
#       [4],
#       [2],
#       [1],
#       [0.]])


#-----------------------------------------------Softmax-----------------------------------------------------
def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z

softmax([9, 8, 11, 10, 8.5]) #or
softmax([[9], [8], [1], [10], [8.5]])
```