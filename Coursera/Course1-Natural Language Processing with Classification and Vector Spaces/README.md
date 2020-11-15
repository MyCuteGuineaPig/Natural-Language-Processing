#### Augmentaion 

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
