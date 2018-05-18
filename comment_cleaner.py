import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer   

#settings
eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
color = sns.color_palette()

APPOS = { "aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "didn't" : "did not", "doesn't" : "does not", "don't" : "do not", "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not", "he'd" : "he would", "he'll" : "he will", "he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is", "it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not", "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "that's" : "that is", "there's" : "there is", "they'd" : "they would", "they'll" : "they will", "they're" : "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not", "we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is", "who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not", "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will", "you're" : "you are", "you've" : "you have", "'re": " are","wasn't": "was not", "we'll":" will", "didn't": "did not"
}

def clean(comment):
    comment = comment.lower()
    # remove new line character
    comment=re.sub('\\n','',comment)
    # remove ip addresses
    comment=re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', comment)
    # remove usernames
    comment=re.sub('\[\[.*\]', '', comment)
    # split the comment into words
    words = tokenizer.tokenize(comment)
    # replace that's to that is by looking up the dictionary
    words=[APPOS[word] if word in APPOS else word for word in words]
    # replace variation of a word with its base form
    words=[lem.lemmatize(word, "v") for word in words]
    # eliminate stop words
    words = [w for w in words if not w in eng_stopwords]
    # now we will have only one string containing all the words
    clean_comment=" ".join(words)
    # remove all non alphabetical characters
    clean_comment=re.sub("\W+"," ",clean_comment)
    clean_comment=re.sub("  "," ",clean_comment)
    return (clean_comment)


