# =============================================================================
# Bidirectional LSTM + Attention — Spam / Ham Classifier
# =============================================================================
# Spam detection can be understood as a form of deceptive intent classification:
# each message carries an illocutionary force — either genuine communication
# (ham) or manipulative persuasion (spam). Recent work in dialogical argument
# mining (Ruiz-Dolz et al., 2024) models illocutionary relations alongside
# argumentative structure, while Freedman & Toni (2024) show that deceptive
# text betrays itself through the quality of its reasoning. Here we take a
# complementary approach: a BiLSTM with additive attention that learns to
# identify tokens most indicative of deceptive communicative intent, allowing
# us to inspect *where* in a message the manipulative signal surfaces.

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


# ── Cell 2: Load & inspect raw data ──────────────────────────────────────────
# The SMS Spam Collection provides a natural testbed for studying deceptive
# vs. genuine communicative intent at the message level.
DATA_PATH = "SPAM text message 20170820 - Data.csv"

raw_df = pd.read_csv(DATA_PATH, encoding="latin-1")

print(f"Raw shape : {raw_df.shape}")
print(f"Columns   : {list(raw_df.columns)}")
print(f"\nNull counts:\n{raw_df.isnull().sum()}")
print(f"\nFirst 5 rows:")
print(raw_df.head())


# ── Cell 3: Clean dataset ───────────────────────────────────────────────────
# Binary encoding: ham (genuine) → 0, spam (deceptive) → 1.
# This mirrors the core distinction in illocutionary act analysis —
# whether a speaker's intent is to inform or to manipulate.
df = raw_df.copy()

null_cols = [col for col in df.columns if df[col].isnull().all()]
if null_cols:
    print(f"Dropping all-null columns: {null_cols}")
    df.drop(columns=null_cols, inplace=True)

df.rename(columns={"Category": "label", "Message": "text"}, inplace=True)
df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

print(f"Cleaned shape : {df.shape}")
print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nEncoded label distribution:")
print(df["label_enc"].value_counts())
print(f"\nSample rows:")
print(df.head(10))


# ── Cell 4: Train/test split & compute text statistics for vectorization ────
# Vocabulary and sequence-length statistics are derived from the training
# partition only, preventing information leakage from evaluation data.
X = df["text"].values
y = df["label_enc"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

word_counts = [len(msg.split()) for msg in X_train]

avg_words_len = int(np.ceil(np.mean(word_counts)))
total_words_length = len(
    set(token for msg in X_train for token in msg.split())
)

print("=" * 50)
print("         Dataset & Text Statistics")
print("=" * 50)
print(f"  Total samples        : {len(X)}")
print(f"  Train samples        : {len(X_train)}")
print(f"  Test samples         : {len(X_test)}")
print(f"  Avg words per SMS    : {avg_words_len}")
print(f"  Min words in a SMS   : {min(word_counts)}")
print(f"  Max words in a SMS   : {max(word_counts)}")
print(f"  Unique vocabulary    : {total_words_length}")
print("=" * 50)


# ── Cell 5: Helper functions & TextVectorization layer ──────────────────────
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compile_and_fit(model, X_tr, y_tr, X_val, y_val, epochs=10, batch_size=32):
    """Compile with BCE + accuracy, train, return history."""
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return history


def get_metrics(model, X_eval, y_true):
    """Predict, threshold at 0.5, print & return acc/prec/rec/f1."""
    y_prob = model.predict(X_eval, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# TextVectorization converts raw strings into padded integer sequences.
from tensorflow.keras.layers import TextVectorization

vectorize_layer = TextVectorization(
    max_tokens=total_words_length,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    output_sequence_length=avg_words_len,
)

vectorize_layer.adapt(X_train)

print(f"TextVectorization configured:")
print(f"  max_tokens             = {total_words_length}")
print(f"  output_sequence_length = {avg_words_len}")
print(f"  Vocabulary size learnt = {vectorize_layer.vocabulary_size()}")
print(f"\nSample vectorization:")
sample = X_train[:2]
for txt, vec in zip(sample, vectorize_layer(sample).numpy()):
    print(f"  \"{txt[:60]}...\"  →  {vec[:15]}...")


# ── Cell 6: Additive Attention layer + BiLSTM-Attention model ────────────────
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    GlobalAveragePooling1D, Flatten, Dense, Dropout, Layer
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# ---------------------------------------------------------------------------
# Additive (Bahdanau) attention over BiLSTM hidden states.
#
# score_t = V^T · tanh(W · h_t + b)          for each timestep t
# alpha   = softmax(scores)                   attention distribution
# context = sum( alpha_t * h_t )              weighted summary
# ---------------------------------------------------------------------------
class AdditiveAttention(Layer):
    """Bahdanau-style single-layer additive attention."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        hidden_dim = input_shape[-1]              # 2 * LSTM_UNITS (bidir concat)
        self.W = self.add_weight(
            name="att_W", shape=(hidden_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_b", shape=(self.units,),
            initializer="zeros", trainable=True,
        )
        self.V = self.add_weight(
            name="att_V", shape=(self.units, 1),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        score = K.tanh(K.dot(hidden_states, self.W) + self.b)
        score = K.dot(score, self.V)
        alpha = K.softmax(score, axis=1)
        context = K.sum(alpha * hidden_states, axis=1)
        return context, K.squeeze(alpha, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# --- Hyperparams ---
EMBEDDING_DIM   = 128
LSTM_UNITS      = 64
ATTENTION_UNITS = 64

# 1. Raw string input
text_input = Input(shape=(1,), dtype=tf.string, name="raw_text")

# 2. Vectorize: string → token IDs
x = vectorize_layer(text_input)

# 3. Embedding: token IDs → dense representations
x = Embedding(
    input_dim=total_words_length,
    output_dim=EMBEDDING_DIM,
    input_length=avg_words_len,
    name="embedding",
)(x)

# 4–5. Stacked BiLSTMs
x = Bidirectional(
    LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_1"
)(x)
x = Bidirectional(
    LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_2"
)(x)

# 6. Attention: compress the full sequence into a fixed-size context vector
context, att_weights = AdditiveAttention(
    units=ATTENTION_UNITS, name="attention"
)(x)

# 7. Classification head
x = Dropout(0.3, name="dropout")(context)
x = Dense(64, activation="relu", name="dense_hidden")(x)
class_output = Dense(1, activation="sigmoid", name="output")(x)

bilstm_model = Model(
    inputs=text_input,
    outputs=[class_output, att_weights],
    name="BiLSTM_Attention_Spam_Classifier",
)
bilstm_model.summary()

bilstm_model.compile(
    optimizer="adam",
    loss={"output": "binary_crossentropy"},
    metrics={"output": "accuracy"},
)

EPOCHS     = 10
BATCH_SIZE = 32

bilstm_history = bilstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

print("\n" + "=" * 50)
print("  Bi-LSTM + Attention — Test Set Metrics")
print("=" * 50)
preds, _ = bilstm_model.predict(X_test, verbose=0)
y_pred = (preds.ravel() >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-score  : {f1_score(y_test, y_pred):.4f}")


# ── Cell 6b: Attention visualisation ─────────────────────────────────────────
import textwrap


def visualize_attention(text: str, ax=None):
    """Heatmap of per-token attention weights for a single SMS."""

    prob, weights = bilstm_model.predict(np.array([text]), verbose=0)
    prob    = prob[0][0]
    weights = weights[0]
    label   = "SPAM" if prob >= 0.5 else "HAM"

    token_ids = vectorize_layer(np.array([text])).numpy()[0]
    vocab     = vectorize_layer.get_vocabulary()
    tokens    = [vocab[tid] if tid < len(vocab) else "<pad>" for tid in token_ids]

    non_pad = [i for i, t in enumerate(tokens) if t not in ("", "<pad>")]
    end     = max(non_pad) + 1 if non_pad else len(tokens)
    tokens  = tokens[:end]
    weights = weights[:end]

    w_min, w_max = weights.min(), weights.max()
    norm_w = (weights - w_min) / (w_max - w_min) if w_max - w_min > 0 else weights

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(len(tokens) * 0.7, 6), 2))
    ax.imshow(norm_w[np.newaxis, :], cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"Prediction: {label} ({prob:.3f})", fontsize=11)
    plt.tight_layout()
    return tokens, norm_w


preds_demo, _ = bilstm_model.predict(X_test, verbose=0)
preds_binary  = (preds_demo.ravel() >= 0.5).astype(int)

correct_spam = [X_test[i] for i in range(len(X_test))
                if y_test[i] == 1 and preds_binary[i] == 1]
correct_ham  = [X_test[i] for i in range(len(X_test))
                if y_test[i] == 0 and preds_binary[i] == 0]

fig, axes = plt.subplots(2, 1, figsize=(14, 5))
visualize_attention(correct_spam[0], ax=axes[0])
visualize_attention(correct_ham[0],  ax=axes[1])
plt.tight_layout()
plt.savefig("attention_visualization.png", dpi=150)
plt.show()


# ── Cell 7: Training curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

loss_key     = "output_loss" if "output_loss" in bilstm_history.history else "loss"
val_loss_key = "val_output_loss" if "val_output_loss" in bilstm_history.history else "val_loss"
acc_key      = "output_accuracy" if "output_accuracy" in bilstm_history.history else "accuracy"
val_acc_key  = "val_output_accuracy" if "val_output_accuracy" in bilstm_history.history else "val_accuracy"

axes[0].plot(bilstm_history.history[loss_key],     label="train")
axes[0].plot(bilstm_history.history[val_loss_key], label="val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(bilstm_history.history[acc_key],     label="train")
axes[1].plot(bilstm_history.history[val_acc_key], label="val")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()


# ── Cell 8: Evaluate on test set (detailed) ─────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred_prob, _ = bilstm_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(cmap="Blues")
plt.title("Bi-LSTM + Attention — Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ── Cell 9: Inference helper ────────────────────────────────────────────────
def predict_message(text: str) -> str:
    """Classify a single SMS and return the predicted intent with confidence."""
    prob, weights = bilstm_model.predict(np.array([text]), verbose=0)
    prob  = prob[0][0]
    label = "spam" if prob >= 0.5 else "ham"
    return f"{label} (confidence: {prob:.4f})"

print(predict_message("You have won a free iPhone! Click here now!"))
print(predict_message("Hey, want to grab coffee after class?"))


# ── Cell 10 (optional): Save / load model ───────────────────────────────────
# bilstm_model.save("bilstm_spam_model.keras")
# loaded_model = tf.keras.models.load_model(
#     "bilstm_spam_model.keras",
#     custom_objects={"AdditiveAttention": AdditiveAttention},
# )
