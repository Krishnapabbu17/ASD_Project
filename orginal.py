#Libararies
#Preprocessing 
import os
import pandas as pd
import numpy as np


#get data
data_path = "C:/Users/pabbu/OneDrive/Autism_Project/Data/Eigsti"
asd_path = os.path.join(data_path, "ASD")
td_path = os.path.join(data_path, "TD")

asd_files = [f for f in os.listdir(asd_path) if f.endswith('.cha.txt')]
td_files = [f for f in os.listdir(td_path) if f.endswith('.cha.txt')]

print("ASD files found: " + str(len(asd_files)))
print("TD files found: " + str(len(td_files)))

#extracting basic features function
def extract_features_from_cha(file_path):
    total_words = 0
    utterance_lengths = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("*CHI:"):
                words = line.strip().split()[1:]  
                total_words += len(words)
                utterance_lengths.append(len(words))

    if utterance_lengths:
        mlu = total_words / len(utterance_lengths)
    else:
        mlu = 0
    return total_words, mlu
    #note: mlu here is mean length of utterance for words/utterances NOT morphemes


#extracting features using extract_feature_from_cha from all files
data=[]
for filename in asd_files:
    file_path = os.path.join(asd_path, filename)
    total_words, mlu = extract_features_from_cha(file_path)
    data.append({"filename": filename, "label": "ASD", "total_words": total_words, "mlu": mlu})

for filename in td_files:
    file_path = os.path.join(td_path, filename)
    total_words, mlu = extract_features_from_cha(file_path)
    data.append({"filename": filename, "label": "TD", "total_words": total_words, "mlu": mlu})


#dataframe
df = pd.DataFrame(data)
print(df.head(30))
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#neural network stuff
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Dropout
#from sklearn.preprocessing import StandardScaler 
    #for training??
#print(tf.__version__)

#Preparing data for model(binary classification??)
df['label_num'] = df['label'].map({'ASD': 1, 'TD': 0})
X = df[['total_words', 'mlu']].values#add more features later(POS,)
y = df['label_num'].values
print(X.shape)
print(y.shape)

#split data into train/test


#additional resources for the basic data preprocessing and features
#https://www.geeksforgeeks.org/machine-learning/data-preprocessing-machine-learning-python/
#https://cookbook.openai.com/examples/chat_finetuning_data_prep


#realization..never cleaned up data, still have to work on that
#I dont think we need SMOTE for Eigsti dataset. If we plan to use more, then might be needed

#stackover flow
