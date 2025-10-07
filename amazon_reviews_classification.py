import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


file_path = "train.ft.txt"
stop_words = set(stopwords.words('english'))
labels, texts = [], []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)  # split once
        if len(parts) == 2:
            labels.append(int(parts[0].replace("__label__", "")))
            texts.append(parts[1])

df = pd.DataFrame({"label": labels, "review": texts})

# 🔹 Keep only 100k rows for faster working
df = df.sample(n=100000, random_state=42).reset_index(drop=True)
df['label'] = df['label'] - 1

corpus = []
ps= PorterStemmer()
x= df.review
y = df.label
y = np.array(y)

for i in range(len(x)):
    review = re.sub('[^a-zA-Z]',' ', x.iloc[i])
    review = review.lower().split()
    review = [ps.stem(words) for words in review if words not in stop_words]
    corpus.append(' '.join(review))


voc_size = 10000
onehot_corp = [one_hot(words, voc_size) for words in corpus]

corp_len = 230
padded_corp = pad_sequences(onehot_corp, padding = 'post', maxlen = corp_len)
print(padded_corp[45])
feature_size = 100
classifier = Sequential()
classifier.add(Embedding(voc_size, feature_size, input_length = corp_len))
classifier.add(LSTM(150))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation = 'sigmoid'))
optimizer = Adam(learning_rate = 0.0001)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
train_x, test_x, train_y, test_y = train_test_split(padded_corp, y, test_size=0.2, random_state=42)

print(classifier.fit(train_x, train_y, epochs = 10, batch_size = 64))