#### Preprocessing 

```python
import nltk                                # Python library for NLP

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
import re
import string

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
def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    """
    Stop words: 
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    """


    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks, hyperlinks 一般出现在twitter最后
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
                                #preserve_case=False：所有变小写, strip_handles=True，去掉@...
                                #reduce_len = False, waaaaayyyy -> waaaaayyyy, 
                                #reduce_len = True, waaaaayyyy -> waaayyy, 

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

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

#''''''''''''''''''''''''''''''''''''''''''''''''reshape''''''''''''''''''''''''''''''''''''''''''''''''''
x2 = np.array([2, 2]).reshape(1, -1) #x2 is [[2 2]]
#-1: numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). 
#It simply means that it is an unknown dimension and we want numpy to figure it out. 

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
FrobeniusNorm = np.linalg.norm(nparray2)  #Get Frobenius norm, return 1 number

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


#'''''''''''''''''''''''''''''''''''''''''''''''Squeeze'''''''''''''''''''''''''''''''''''''''''''''''''
#make to single-dimensional entries from the shape of an array.

x = np.array([[[0], [1], [2]]])
x.shape
#(1, 3, 1)

np.squeeze(x).shape
#(3,)

print(np.squeeze(x)) #array([0, 1, 2])

#'''''''''''''''''''''''''''''''''''''''''''''''Keep_dims'''''''''''''''''''''''''''''''''''''''''''''''''
print_matrix(transition_matrix)
"""
        NN     RB     TO
NN  1624.1  243.1  525.6
RB    35.8  226.3   85.5
TO    73.4   20.0    0.2
"""
# Compute sum of row for each row
rows_sum = transition_matrix.sum(axis=1, keepdims=True)
# 3 by 1 matrix
#array([[2392.8],
#       [ 347.6],
#       [  93.6]])

# Without, keepdims=True, will be (3,) vector


#'''''''''''''''''''''''''''''''''''''''''''''''Cosine Similarity'''''''''''''''''''''''''''''''''''''''''''''''''
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot/(norma*normb)
    return cos

#'''''''''''''''''''''''''''''''''''''''''''''''Euclidean distance'''''''''''''''''''''''''''''''''''''''''''''''''
def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """
    d = np.linalg.norm(A-B)
    return d

#'''''''''''''''''''''''''''''''''''''''''''''''Size Broadcast'''''''''''''''''''''''''''''''''''''''''''''''''
a = array([1, 2, 3]) # array([1, 2, 3])
a[None,:]  # array([[1, 2, 3]])
```

## PCA

```python

#'''''''''''''''''''''''''''''''''''''''''''''''PCA'''''''''''''''''''''''''''''''''''''''''''''''''
# Method 1: 
from sklearn.decomposition import PCA      # PCA library

x = x - np.mean(x) # Center x. Remove its mean, numpy array
y = y - np.mean(y) # Center y. Remove its mean, numpy array
data = pd.DataFrame({'x': x, 'y': y}) # Create a data frame with x and y

pca = PCA(n_components=2) # Instantiate a PCA. Choose to get 2 output variables, 
                          #n_components : number of components to keep
pcaTr = pca.fit(data)

rotatedData = pcaTr.transform(data) # Transform the data base on the rotation matrix of pcaTr
# # Create a data frame with the new variables. We call these new variables PC1 and PC2
dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2']) 

pcaTr.components_ #'Eigenvectors or principal component: 
pcaTr.explained_variance_ #Eigenvalues or explained variance


#Method 2:

def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data

    Note: np.linalg.eigh guarantees you that the eigenvalues are sorted and uses a faster algorithm that takes advantage of the fact 
        that the matrix is symmetric. If you know that your matrix is symmetric, use this function.
    Attention, eigh doesn't check if your matrix is indeed symmetric, it by default just takes the lower triangular part of the matrix 
    and assumes that the upper triangular part is defined by the symmetry of the matrix.
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # mean center the data
    X_demeaned = X - np.mean(X, axis=0)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    #If rowvar = True, then each row represents a variable, with observations in the columns. 
    #   Otherwise, ach column represents a variable, while the rows contain observations.

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    
    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:,:n_components]

    # transform the data by multiplying the transpose of the eigenvectors 
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(X_demeaned,eigen_vecs_subset)
    return X_reduced
```