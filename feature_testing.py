
        #Importing Libraries
import os
import pandas as pd
import re
import numpy as np
#import seaborn as sns
import pymc as pm
import matplotlib.pyplot as plt
import tensorflow as tf
import powerlaw
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
#Feature extraction functions (same as before)
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

def extract_zipf_features_from_text(text):
    # Tokenize and clean text (simple whitespace split)
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        # Avoid errors on empty input
        return {
            "zipf_xmin": 0,
            "zipf_alpha": 0,
            "zipf_pvalue": 0,
            "zipf_likelihood_ratio": 0,
            "zipf_likelihood_ratio_significance": 0,
        }

    # Count word frequencies
    freqs = list(Counter(words).values())

    # 🚨 New: skip if not enough unique word frequencies
    if len(freqs) < 5:
        return {
            "zipf_xmin": 0,
            "zipf_alpha": 0,
            "zipf_pvalue": 0,
            "zipf_likelihood_ratio": 0,
            "zipf_likelihood_ratio_significance": 0,
        }

    np.random.seed(42)
    
    # Fit power law distribution
    fit = powerlaw.Fit(freqs, discrete=True)

    xmin = fit.xmin
    alpha = fit.alpha

    # Bootstrap p-value for goodness of fit
    pvalue = fit.distribution_compare('power_law', 'lognormal')[1]

    # Likelihood ratio and significance for power-law vs lognormal
    R, p = fit.distribution_compare('power_law', 'lognormal')

    return {
        "zipf_xmin": xmin,
        "zipf_alpha": alpha,
        "zipf_pvalue": pvalue,
        "zipf_likelihood_ratio": R,
        "zipf_likelihood_ratio_significance": p,
    }

def extract_features_from_cha(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Extract usual features
    features = extract_features_from_lines(lines)

    # Get full child speech text (joining *CHI: lines)
    child_speech = " ".join(
        re.sub(r'[^\w\s-]', '', line[5:].lower())
        for line in lines if line.startswith("*CHI:")
    )

    # Extract Zipf features
    zipf_features = extract_zipf_features_from_text(child_speech)

    # Combine
    features.update(zipf_features)
    return features

if __name__ == "__main__":

    #Load data
    data_path = "C:/Users/pabbu/OneDrive/Autism_Project/Data/Eigsti"
    asd_path = os.path.join(data_path, "ASD")
    td_path = os.path.join(data_path, "TD")

    asd_files = [f for f in os.listdir(asd_path) if f.endswith(".cha.txt")]
    td_files = [f for f in os.listdir(td_path) if f.endswith(".cha.txt")]

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

    #DataFrame
    df = pd.DataFrame(data)
    df['label_num'] = df['label'].map({'ASD': 1, 'TD': 0})
    print(df.shape)
    # Defining Features
    feature_cols = [
    'mluw', 'mlum', 'mlt_ratio',
    'pronouns_per_100w', 'conjunctions_per_100w', 'negations_per_100w',
    'type_token_ratio', 'repetitions',
    'total_words',
    'zipf_xmin', 'zipf_alpha', 'zipf_pvalue',
    'zipf_likelihood_ratio', 'zipf_likelihood_ratio_significance',
    ]



    #feature_cols = ['mluw', 'mlum', 'mlt_ratio', 'pronouns', 'conjunctions', 'negations', 'total_words']
    #feature_cols = ['mluw', 'mlum', 'pronouns','conjunctions','negations', 'total_words']

    #cLEANING
    # Drop rows with missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    #Remove any rows with 0 utterances (shows invalid mluw/mlum)
    df = df[df['total_words'] > 0]
    df = df[df['mluw'] > 0]
    df = df[df['mlum'] > 0]
    df = df[df['mlt_ratio'] > 0]

    #Remove extreme outliers (Z-score > 3)
    from scipy.stats import zscore
    df_z = df[feature_cols].apply(zscore)
    df = df[(np.abs(df_z) < 3).all(axis=1)]

    #Reset index after cleaning
    df.reset_index(drop=True, inplace=True)


    # Select features
    X = df[feature_cols].values
    y = df['label_num'].values

    # Train/Test Split (❗ split first)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale (✅ now fit only on training set, then transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    from scipy.stats import ttest_ind

    for feature in feature_cols:
        asd_vals = df[df['label_num'] == 1][feature]
        td_vals = df[df['label_num'] == 0][feature]
        t_stat, p_val = ttest_ind(asd_vals, td_vals)
        print(f"{feature}: t = {t_stat:.3f}, p = {p_val:.4f}")

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for i in indices:
        print(f"{feature_cols[i]}: {importances[i]:.4f}")


