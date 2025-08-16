# 1. Libraries and Load Data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
#print(df.head(30))
#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())

#neural network stuff
#libraries
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Dropout
#from sklearn.preprocessing import StandardScaler 
    #for training??
#print(tf.__version__)

#Preparing data for model(binary classification??)
df['label_num'] = df['label'].map({'ASD': 1, 'TD': 0})
print(df.head())
X = df[['total_words', 'mlu']].values#add more features later(POS,)
y = df['label_num'].values
#print(X.shape)
#print(y.shape)

#Scalinf
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Shapes:", X_train.shape, X_test.shape)

#tutorial stuff(model 4)
tf.random.set_seed(42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, verbose=0)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nModel loss on test: {loss:.4f}')
print(f'Model accuracy on test: {accuracy*100:.2f}%')


# Decision Boundary Plot

def plot_decision_boundary(model, X, y):
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] 
  # Make predictions using the trained model
  y_pred = model.predict(x_in)
  # Check for multi-class
  if len(y_pred[0]) > 1:
     print("doing multiclass classification...")
     # We have to reshape our predictions to get them ready for plotting
     y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
     print("doing binary classifcation...")
     y_pred = np.round(y_pred).reshape(xx.shape)
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())


# Plot train vs test decision boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()









#additional resources for the basic data preprocessing and features
#https://www.geeksforgeeks.org/machine-learning/data-preprocessing-machine-learning-python/
#https://cookbook.openai.com/examples/chat_finetuning_data_prep


#realization..never cleaned up data, still have to work on that
#I dont think we need SMOTE for Eigsti dataset. If we plan to use more, then might be needed

#stackover flow
