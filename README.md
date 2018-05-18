# LSTM-Classification 🧠

Given a dataset of 160,000 comments from Wikipedia's talk page edits, we aim to analyse this data and model a classifier by which we can classify comments based on their level and type of toxicity. Each comment within the train file is loaded with an `id` and the following 6 binary labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`; each of which may have value of either `0` or `1`. 

Here you you can find some datails about steps taken to make model with **96.7% accuracy** using **[Keras](https://keras.io/)**.


## Table of Contents
  1. [Usage](#usage)
  1. [Data Exploration](#data-exploration)
  1. [Preprocessing](#preprocessing)
  1. [Training the model using Keras](#training-the-model-using-keras)
  1. [Results](#results)

## Usage
Clone the repo:
```
git clone https://github.com/iamhosseindhv/LSTM-Classification.git lstm
cd lstm
```
In order to run the scripts, you have to have datasets downloaded. You can login to your Kaggle account, download the following files, and put them in the same directory you have cloned this repo in:
* [training dataset](https://www.kaggle.com/c/8076/download/train.csv.zip)
* [test dataset](https://www.kaggle.com/c/8076/download/test.csv.zip)
* [sample submission file](https://www.kaggle.com/c/8076/download/sample_submission.csv.zip)

Now you can run the script:
```
python lstm_classifier.py
```

**Prerequisites:** In case you want to clone the repo and play with stuff, you need the following installed:
* [TensorFlow](https://www.tensorflow.org/install/)
* [Keras](https://keras.io/#installation)
* [Pandas](https://pandas.pydata.org)
* Numpy

## Data Exploration
First, let's take a look at our train file: (sorry for the bad language 🤨)
```
id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
00190820581d90ce,"F**K YOUR F****Y MOTHER IN THE A**, DRY!",1,0,1,0,1,0
00229d44f41f3acb,"Locking this page would also violate WP:NEWBIES.  Whether you like it or not, conservatives are Wikipedians too.",0,0,0,0,0,0
0001d958c54c6e35,"You, sir, are my hero. Any chance you remember what page that's on?",0,0,0,0,0,0
...
```
The point of exploration is to get some insight into our data. We demonstrate the distribution of each class over the whole dataset. Comments which do not have a `1` in  any of the toxicity classes are considered as clean in this dataset.

```python
# Complete code at plots/plot_balance.py

column = train.iloc[:,2:].sum()
# plot
plt.figure(figsize=(8,4))
ax = sns.barplot(column.index, column.values, alpha=0.8)
plt.title("Balance of classes")
plt.ylabel('Number of occurences', fontsize=13)
plt.xlabel('Quantity per class', fontsize=13)
# adding the text labels
rects = ax.patches
labels = column.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()
```
<p align="center">
  <img width="600" alt="balance of classes" src="http://i.imgur.com/cFlwyJx.png">
</p>

</br>

As we can see, if we sum up the number on top of each of the bars, it will exceed the total number of comments. Therefore, there must be cases where a comment is classified as severe_toxic AND threat, say.

```python
# Complete code at plots/plot_tags.py

rowsums=train.iloc[:,2:].sum(axis=1)
x = rowsums.value_counts()
# plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Multiple tags per comment")
plt.ylabel('Number of occurrences', fontsize=13)
plt.xlabel('Quantity per tag', fontsize=13)
# adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()
```
<p align="center">
  <img width="700" alt="tags per comment" src="http://i.imgur.com/cAr6YRR.png">
</p>

Besides, we count number of unique words in each class, and results are shown on the following table (figure below). Interestingly, number of unique words in clean comment is almost double of those in other toxicity classes. From these findings we can conclude that toxic (i.e. unclean) comments are likely to include spam..
<p align="center">
  <img width="600" alt="average unique words in class" src="http://i.imgur.com/vuQZ02t.png">
</p>


</br>

## Preprocessing
The raw data cannot be used as is, since it has defects which may lead to **overfitting** and consequently an inaccurate model. Therefore, it needs to be processed and cleaned first. 

The python script below, identifies and removes unnecessary noise. These include: **stop-words**, IP addresses, usernames, new line characters, punctuations, quote marks. In addition to these, using a dictionary in our script, we replace contracted words (e.g. you’re) with their original form (i.e. you are in this case).

Another important step in stemming is **lemmatising**. We make use of `WordNetLemmatizer` from `nltk.stem.wordnet` to replace all variations of a word with its base form (e.g. leaves will become leaf after this operation). Here's an example of a comment BEFORE and AFTER cleaning step:

```
BEFORE:
Hi! I'm back again [[User :Spinningspark|Spark]] ! Can you prove it isn't ? You must not play Metal 
Gear Solid 2 that often !!! Playing is not good :)
Last warning!
Stop undoing my edits or die! .80.80.120.181

AFTER:
hi i am back prove is not must play metal gear solid 2 often play good last warn stop undo edit die
```
Python script:
```python
# Complete code at comment_cleaner.py

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
```

</br>

## Training the model using Keras

```python
# Complete code at lstm_classifier.py

# read and clean comments
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"].apply(lambda comment: clean(comment))
list_sentences_test = test["comment_text"].apply(lambda comment: clean(comment))

# maximum number of unique words in our dictionary
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

# index representation of the comments
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# in the data exploration step, I found the average length of comments,
# and setting maximum length to 100 seemed to be reasonable.
maxlen = 100
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

# first step of making the model
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

# size of the vector space
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

output_dimention = 60
x = LSTM(output_dimention, return_sequences=True,name='lstm_layer')(x)
# reduce dimention
x = GlobalMaxPool1D()(x)
# disable 10% precent of the nodes
x = Dropout(0.1)(x)
# pass output through a RELU function
x = Dense(50, activation="relu")(x)
# another 10% dropout
x = Dropout(0.1)(x)
# pass the output through a sigmoid layer, since 
# we are looking for a binary (0,1) classification 
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
# we use binary_crossentropy because of binary classification
# optimise loss by Adam optimiser
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# make predictions and writing them in the file predictions.csv
y_pred = model.predict(X_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('predictions.csv', index=False)
```

</br>

## Results
Here is the final prediction file outputted by our model. It gives a number between `0` and `1` for each class.

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12, 0.9617284536361694, 0.22618402540683746, 0.8573611974716187, 0.061656974256038666, 0.7879733443260193, 0.18741782009601593
0000247867823ef7, 0.004322239197790623, 0.4489359424915165e-05, 0.0006502021569758654, 0.536748747341335e-05, 0.0004672454087994993, 0.00010046786337625235
00013b17ad220c46, 0.007684065029025078, 0.666447690082714e-05, 0.000907833396922797, 0.4642296163365245e-05, 0.000789982674177736, 0.0001430132397217676
...
```
Since this was a part of [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), I submitted the results to Kaggle, and our model got the following score:

<p align="center">
  <img alt="kaggle score for the model" src="http://i.imgur.com/eLvbQSs.png">
</p>



