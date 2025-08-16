import os
import pandas as pd
import re
import numpy as np
import pymc as pm
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from scipy.stats import zscore
import matplotlib.pyplot as plt

# --- Feature Extraction ---
def extract_features_from_lines(lines):
    pronouns = {'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers',
                'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself',
                'ourselves', 'yourselves', 'themselves', 'who', 'whom', 'whose', 'which',
                'that', 'anyone', 'someone', 'everyone', 'somebody', 'anybody',
                'everybody', 'each', 'either', 'neither', 'this', 'these', 'those'}

    conjunctions = {'and', 'but', 'or', 'because', 'so', 'if', 'although', 'while', 'when',
                    'nor', 'yet', 'unless', 'until', 'even though', 'as', 'as if',
                    'as though', 'whereas', 'since', 'whether', 'before', 'after',
                    'once', 'than', 'though', 'provided', 'lest', 'in case', 'now that',
                    'even if', 'rather than', 'both', 'either', 'neither', 'not only',
                    'just as'}

    negations = {'no', 'not', "n't", 'never', 'nothing', 'nobody', 'nowhere', 'neither',
                 'no one', 'none', 'without', 'hardly', 'scarcely', 'barely', 'cannot',
                 "can't", "don't", "won't", "isn't", "wasn't", "weren't", "didn't",
                 "hasn't", "haven't", "hadn't", "wouldn't", "shouldn't", "mustn't",
                 "mightn't"}

    child_word_count = 0
    child_morpheme_count = 0
    child_utterance_count = 0
    pronoun_count = 0
    conj_count = 0
    negation_count = 0
    mother_utterance_count = 0
    type_token_set = set()
    repetition_count = 0

    for line in lines:
        line = line.strip()
        if line.startswith("*CHI:"):
            utterance = re.sub(r'[^\w\s-]', '', line[5:].lower())
            words = utterance.split()
            morphemes = sum(len(word.split('-')) for word in words)

            child_word_count += len(words)
            child_morpheme_count += morphemes
            child_utterance_count += 1

            type_token_set.update(words)

            pronoun_count += sum(1 for word in words if word in pronouns)
            conj_count += sum(1 for word in words if word in conjunctions)
            negation_count += sum(1 for word in words if word in negations)

            # Count repeated words (disfluency)
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    repetition_count += 1

        elif any(line.startswith(speaker) for speaker in ["*MOT:", "*INV:", "*INV1:", "*INV2:"]):
            mother_utterance_count += 1

    mluw = child_word_count / child_utterance_count if child_utterance_count else 0
    mlum = child_morpheme_count / child_utterance_count if child_utterance_count else 0
    mlt_ratio = child_utterance_count / mother_utterance_count if mother_utterance_count else 0
    ttr = len(type_token_set) / child_word_count if child_word_count else 0

    return {
        "mluw": mluw,
        "mlum": mlum,
        "mlt_ratio": mlt_ratio,
        "pronouns_per_100w": 100 * pronoun_count / child_word_count if child_word_count else 0,
        "conjunctions_per_100w": 100 * conj_count / child_word_count if child_word_count else 0,
        "negations_per_100w": 100 * negation_count / child_word_count if child_word_count else 0,
        "type_token_ratio": ttr,
        "repetitions": repetition_count,
        "total_words": child_word_count,
    }

def extract_features_from_cha(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    features = extract_features_from_lines(lines)
    return features

if __name__ == "__main__":
    # Data paths
    data_path = "C:/Users/pabbu/OneDrive/Autism_Project/Data/Eigsti"
    asd_path = os.path.join(data_path, "ASD")
    td_path = os.path.join(data_path, "TD")

    # List files
    asd_files = [f for f in os.listdir(asd_path) if f.endswith(".cha.txt")]
    td_files = [f for f in os.listdir(td_path) if f.endswith(".cha.txt")]

    # Extract features
    data = []
    for filename in asd_files:
        file_path = os.path.join(asd_path, filename)
        features = extract_features_from_cha(file_path)
        features.update({"filename": filename, "label": "ASD"})
        data.append(features)

    for filename in td_files:
        file_path = os.path.join(td_path, filename)
        features = extract_features_from_cha(file_path)
        features.update({"filename": filename, "label": "TD"})
        data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data)
    df['label_num'] = df['label'].map({'ASD': 1, 'TD': 0})

    # Select features as requested
    feature_cols = [
        'mlum',                 # MLU: mean length of utterance (morphemes)
        'mlt_ratio',            # Mean length turn ratio
        'pronouns_per_100w',    # Percent pronouns
        'conjunctions_per_100w',# Percent conjunctions
        'negations_per_100w',   # Percent negations
        'total_words'           # Number of words spoken
    ]

    # Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df['total_words'] > 0]
    df = df[df['mlum'] > 0]
    df = df[df['mlt_ratio'] > 0]

    df_z = df[feature_cols].apply(zscore)
    df = df[(np.abs(df_z) < 3).all(axis=1)]
    df.reset_index(drop=True, inplace=True)

    # Prepare X and y
    X = df[feature_cols].values
    y = df['label_num'].values

    accuracies = []

    for run in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_features = X_train_scaled.shape[1]

        with pm.Model() as model:
            mus = pm.Normal('mus', mu=0, sigma=1, shape=n_features)
            sigmas = pm.HalfNormal('sigmas', sigma=1, shape=n_features)
            pm.Normal('obs', mu=mus, sigma=sigmas, observed=X_train_scaled)
            trace = pm.sample(2000, tune=1000, target_accept=0.9, chains=2, progressbar=False, random_seed=run)

        mus_samples = trace.posterior['mus'].stack(sample=("chain", "draw")).values.T
        sigmas_samples = trace.posterior['sigmas'].stack(sample=("chain", "draw")).values.T

        n_synthetic = 1000
        synthetic_features = []
        for _ in range(n_synthetic):
            idx = np.random.randint(mus_samples.shape[0])
            mu = mus_samples[idx]
            sigma = sigmas_samples[idx]
            sample = np.random.normal(mu, sigma)
            synthetic_features.append(sample)
        synthetic_features = np.array(synthetic_features)

        proportion_asd = np.mean(y_train)
        synthetic_labels = np.random.choice([1, 0], size=n_synthetic, p=[proportion_asd, 1 - proportion_asd])

        X_train_aug = np.vstack([X_train_scaled, synthetic_features])
        y_train_aug = np.hstack([y_train, synthetic_labels])

        tf.random.set_seed(run)
        nn_model = Sequential([
            tf.keras.Input(shape=(X_train_aug.shape[1],)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        nn_model.compile(loss='binary_crossentropy',
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        nn_model.fit(X_train_aug, y_train_aug, epochs=30, validation_split=0.2,
                     callbacks=[early_stop], verbose=0)

        loss, accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
        accuracies.append(accuracy)

    print(f"Min accuracy: {min(accuracies):.4f}")
    print(f"Max accuracy: {max(accuracies):.4f}")
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")

    # Plot accuracy over runs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), accuracies, marker='o', linestyle='-', color='blue')
    plt.title("Accuracy over 100 Runs")
    plt.xlabel("Run #")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
