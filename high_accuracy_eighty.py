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

    print(f"Data shape after cleaning: {df.shape}")
    print(df.info())
    print(df.describe())
    print(df.head())

    #Select features
    #feature_cols = ['mluw', 'mlum', 'mlt_ratio', 'pronouns', 'conjunctions', 'negations', 'total_words']
    X = df[feature_cols].values
    y = df['label_num'].values

    #Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Shapes:", X_train.shape, X_test.shape)

    # 🔁 MCMC Augmentation
    n_features = X_train.shape[1]


    with pm.Model() as model:
        mus = pm.Normal('mus', mu=0, sigma=1, shape=n_features)
        sigmas = pm.HalfNormal('sigmas', sigma=1, shape=n_features)
        obs = pm.Normal('obs', mu=mus, sigma=sigmas, observed=X_train)

        trace = pm.sample(2000, tune=1000, target_accept=0.9, chains=2, random_seed=42)

    # Extract samples for mus and sigmas
    mus_samples = trace.posterior['mus'].stack(sample=("chain", "draw")).values  # shape (n_features, n_samples)
    sigmas_samples = trace.posterior['sigmas'].stack(sample=("chain", "draw")).values  # same shape

    # Transpose to (n_samples, n_features)
    mus_samples = mus_samples.T
    sigmas_samples = sigmas_samples.T

    # Draw synthetic samples
    n_synthetic = 500
    synthetic_features = []
    for _ in range(n_synthetic):
        idx = np.random.randint(mus_samples.shape[0])
        mu = mus_samples[idx]
        sigma = sigmas_samples[idx]
        sample = np.random.normal(mu, sigma)
        synthetic_features.append(sample)

    synthetic_features = np.array(synthetic_features)
    # Assign synthetic labels
    proportion_asd = np.mean(y_train)
    synthetic_labels = np.random.choice([1, 0], size=n_synthetic, p=[proportion_asd, 1 - proportion_asd])

    # Combine with real data
    X_train_aug = np.vstack([X_train, synthetic_features])
    y_train_aug = np.hstack([y_train, synthetic_labels])
   # Model architecture
    tf.random.set_seed(42)

    nn_model = Sequential([
        tf.keras.Input(shape=(X_train_aug.shape[1],)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    nn_model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        metrics=['accuracy']
    )

    # Add early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    nn_model.fit(
        X_train_aug, y_train_aug,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    #Evaluation
    loss, accuracy = nn_model.evaluate(X_test, y_test)
    print(f'\nModel loss on test: {loss:.4f}')
    print(f'Model accuracy on test: {accuracy*100:.2f}%')

    #Plotting Decision Boundary (CHAT GPT)
    def plot_decision_boundary(model, X, y, x_index=0, y_index=1):
        x_min, x_max = X[:, x_index].min() - 0.1, X[:, x_index].max() + 0.1
        y_min, y_max = X[:, y_index].min() - 0.1, X[:, y_index].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
        x_in = np.c_[xx.ravel(), yy.ravel()]
    
        full_input = np.zeros((x_in.shape[0], X.shape[1]))
        full_input[:, x_index] = x_in[:, 0]
        full_input[:, y_index] = x_in[:, 1]

        y_pred = nn_model.predict(full_input)
        y_pred = np.round(y_pred).reshape(xx.shape)

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, x_index], X[:, y_index], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlabel(feature_cols[x_index])
        plt.ylabel(feature_cols[y_index])

    #Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(nn_model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(nn_model, X_test, y_test)
    plt.tight_layout()
    plt.show()



