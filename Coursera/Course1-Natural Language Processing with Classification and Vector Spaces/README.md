#### Preprocessing 

```python
import nltk                                # Python library for NLP

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


# instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)  #preserve_case=False：所有变小写, strip_handles=True，去掉@...
                                #reduce_len = False, waaaaayyyy -> waaaaayyyy, 
                                #reduce_len = True, waaaaayyyy -> waaayyy, 

# tokenize tweets
tweet_tokens = tokenizer.tokenize(tweet2) #out is list

#stop word
stopwords_english = stopwords.words('english') 


# Instantiate stemming class 
# happy, happiness, happier -> happi
stemmer = PorterStemmer() 
stem_word = stemmer.stem(word)  # stemming word


#Or use utils, same as 
# 1. Remove hyperlinks, Twitter marks and styles
# 2. Tokenize the string
# 3. Remove stop words and punctuations
# 4. Stemming
from utils import process_tweet # Import the process_tweet function

tweets_stem = process_tweet(tweet); # Preprocess a given tweet

```

#### Numpy Array

Note:  the transpose operation does not affect 1D arrays. axis = 0, do for every column, axis = 1, do for every row

```Python
alist = [1, 2, 3, 4, 5]   
narray = np.array([1, 2, 3, 4])

print(narray + narray) #[2 4 6 8]
print(alist + alist) #[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

print(narray * 3) #[ 3  6  9 12]
print(alist * 3) #[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

okmatrix = np.array([[1, 2], [3, 4]]) 
print(okmatrix * 2) # [[2, 4], [6, 8]]

result = okmatrix * okmatrix # Multiply each element by itself
print(result) #[[ 1  4],[ 9 16]]

badmatrix = np.array([[1, 2], [3, 4], [5, 6, 7]]) # Define a matrix. Note the third row contains 3 elements
print(badmatrix * 2) # It is supposed to scale the whole matrix
                    #[list([1, 2, 1, 2]) list([3, 4, 3, 4]) list([5, 6, 7, 5, 6, 7])]

#''''''''''''''''''''''''''''''''''''''''''''''''Transpose''''''''''''''''''''''''''''''''''''''''''''''''''

#However, note that the transpose operation does not affect 1D arrays.
nparray = np.array([1, 2, 3, 4]) 
print(nparray)  #[1 2 3 4]
print(nparray.T) #[1 2 3 4]

nparray = np.array([[1, 2, 3, 4]])
print(nparray)  #[[1 2 3 4]]
print(nparray.T)  # [[1]
                  #  [2]
                  #  [3]
                  #  [4]]


#''''''''''''''''''''''''''''''''''''''''''''''''Norm'''''''''''''''''''''''''''''''''''''''''''''''''''''''

#get the norm by rows or by columns:
#axis=0 means get the norm of each column
# axis=1 means get the norm of each row.

nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) # Define a 3 x 2 matrix. 

normByCols = np.linalg.norm(nparray2, axis=0) # Get the norm for each column. Returns 2 elements
normByRows = np.linalg.norm(nparray2, axis=1) # get the norm for each row. Returns 3 elements

print(normByCols) #[3.74165739 3.74165739]
print(normByRows) #[1.41421356 2.82842712 4.24264069]


#'''''''''''''''''''''''''''''''''''''''''''''''Dot Product'''''''''''''''''''''''''''''''''''''''''''''''''

#We strongly recommend using np.dot, since it is the only method that accepts arrays and lists without problems

nparray1 = np.array([0, 1, 2, 3]) # Define an array
nparray2 = np.array([4, 5, 6, 7]) # Define an array

flavor1 = np.dot(nparray1, nparray2) # Recommended way
print(flavor1)  #38

flavor2 = np.sum(nparray1 * nparray2) # Ok way
print(flavor2)  #38

flavor3 = nparray1 @ nparray2         # Geeks way
print(flavor3)  #38

# As you never should do:             # Noobs way
flavor4 = 0
for a, b in zip(nparray1, nparray2):
    flavor4 += a * b
    
print(flavor4)  #38

```