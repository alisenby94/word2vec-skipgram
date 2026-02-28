#!/usr/bin/env python
"""
Skip-gram Word2Vec Implementation with Negative Sampling

Implements the Skip-gram Word2Vec model from scratch on WikiText-103,
including word similarity evaluation and word analogy tasks.

Parts:
- Part 1: Implement Skip-gram with Negative Sampling on WikiText-103
- Part 2: Find similar words using cosine similarity
- Part 3: Solve word analogies using GloVe vectors
"""


import re
import json
import math
import random
import collections
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

import matplotlib.pyplot as plt
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Setup and Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Load and Preprocess Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Loading WikiText-103 ===")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset["train"]["text"]
print(f"Raw lines: {len(train_data):,}")


def preprocess_line(line: str) -> list[str]:
    """Lowercase, tokenize, and filter tokens.
    
    Keeps only tokens that:
    - are at least 2 characters long
    - contain at least one alphabetic character
    """
    return [t for t in re.findall(r'\w+', line.lower())
            if len(t) >= 2 and any(c.isalpha() for c in t)]


print("\nTokenizing corpus...")
corpus: list[list[str]] = []
for line in tqdm(train_data, desc="Tokenizing", unit=" lines"):
    tokens = preprocess_line(line)
    if tokens:
        corpus.append(tokens)

total_tokens = sum(len(s) for s in corpus)
print(f"Sentences: {len(corpus):,}")
print(f"Total tokens: {total_tokens:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Build Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

MIN_FREQ = 50
UNK = "<UNK>"

word_freq: collections.Counter = collections.Counter(
    token for sentence in corpus for token in sentence
)
print(f"\nRaw vocabulary size: {len(word_freq):,}")

vocab_words = [UNK] + sorted(w for w, c in word_freq.items() if c >= MIN_FREQ)
word_to_idx: dict[str, int] = {w: i for i, w in enumerate(vocab_words)}
idx_to_word: dict[int, str] = {i: w for w, i in word_to_idx.items()}
VOCAB_SIZE = len(vocab_words)

print(f"Filtered vocabulary size (freq ≥ {MIN_FREQ}): {VOCAB_SIZE:,}")


def encode_sentence(tokens: list[str]) -> list[int]:
    """Convert tokens to indices."""
    return [word_to_idx.get(t, word_to_idx[UNK]) for t in tokens]


encoded_corpus: list[list[int]] = [encode_sentence(s) for s in corpus]


# ─────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 5

def generate_skipgram_pairs(
    encoded_corpus: list[list[int]], window: int
) -> list[tuple[int, int]]:
    """Return (target_idx, context_idx) pairs for the full corpus."""
    pairs = []
    for sentence in tqdm(encoded_corpus, desc="Skip-gram pairs", unit=" sents", leave=False):
        n = len(sentence)
        for i, target in enumerate(sentence):
            lo = max(0, i - window)
            hi = min(n - 1, i + window)
            for j in range(lo, hi + 1):
                if j != i:
                    pairs.append((target, sentence[j]))
    return pairs


# NOTE: materialising all ~834M pairs for WikiText-103 requires ~50 GB RAM.
# We generate a tiny sample for display only; training streams pairs on-the-fly.
print("Sampling skip-gram pairs from first 100 sentences for display ...")
sample_pairs = generate_skipgram_pairs(encoded_corpus[:100], WINDOW_SIZE)
est_total = total_tokens * 2 * WINDOW_SIZE
print(f"Estimated total skip-gram pairs (full corpus): ~{est_total:,}")
print(f"First 5 pairs (indices): {sample_pairs[:5]}")
print(f"First 5 pairs (words)  : {[(idx_to_word[t], idx_to_word[c]) for t, c in sample_pairs[:5]]}")


# ─────────────────────────────────────────────────────────────────────────
NUM_NEGATIVES = 10

# Build noise distribution: freq^(3/4), normalised
freq_array = np.array(
    [word_freq.get(idx_to_word[i], 0) ** 0.75 for i in range(VOCAB_SIZE)],
    dtype=np.float32,
)
# <UNK> should not be sampled as a negative
freq_array[word_to_idx[UNK]] = 0.0

# Pre-load the weights tensor onto the GPU once — torch.multinomial samples
# directly on-device, saving the ~10-15 ms np.random.choice call per batch.
noise_weights = torch.from_numpy(freq_array).to(DEVICE)  # (V,), float32


def sample_negatives(batch_size: int, k: int) -> torch.Tensor:
    """
    Sample k negatives for each item in the batch entirely on-device.
    Samples batch_size*k indices in one 1-D multinomial call then reshapes —
    avoids materialising a (batch_size, vocab_size) weight matrix.
    Returns shape (batch_size, k) on DEVICE.
    """
    return torch.multinomial(
        noise_weights,
        num_samples=batch_size * k,
        replacement=True,
    ).view(batch_size, k)  # (batch_size, k)


print(f"Noise distribution built over {VOCAB_SIZE:,} words (k={NUM_NEGATIVES} negatives).")
# Sanity check: top-5 words by noise weight
top5 = np.argsort(freq_array)[::-1][:5]
print("Top-5 most likely negatives:", [idx_to_word[i] for i in top5])


# ─────────────────────────────────────────────────────────────────────────
EMBED_DIM = 300


class SkipGramNegSampling(nn.Module):
    """
    Skip-gram model with Negative Sampling.

    Two separate embedding tables:
        W_in  — target (centre) word embeddings  [V x E]
        W_out — context (output) word embeddings [V x E]
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Initialise uniformly in (-0.5/E, 0.5/E) following original C implementation
        bound = 0.5 / embed_dim
        self.W_in = nn.Embedding(vocab_size, embed_dim)
        self.W_out = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.W_in.weight, -bound, bound)
        nn.init.uniform_(self.W_out.weight, -bound, bound)  # same as W_in — avoids cold-start

    def forward(
        self,
        target_ids: torch.Tensor,        # (B,)
        pos_context_ids: torch.Tensor,   # (B,)
        neg_context_ids: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        """
        Returns mean negative-sampling loss over the batch.
        """
        v_c = self.W_in(target_ids)          # (B, E)
        u_o = self.W_out(pos_context_ids)    # (B, E)
        u_k = self.W_out(neg_context_ids)    # (B, K, E)

        # Positive score: dot product, then log-sigmoid
        pos_score = torch.sum(v_c * u_o, dim=1)          # (B,)
        pos_loss  = -torch.nn.functional.logsigmoid(pos_score)

        # Negative scores: (B, K) dot products
        neg_score = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss  = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)  # (B,)

        return (pos_loss + neg_loss).mean()

    def get_embeddings(self) -> np.ndarray:
        """Return W_in weight matrix as a NumPy array."""
        return self.W_in.weight.detach().cpu().numpy()


model = SkipGramNegSampling(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")


# ─────────────────────────────────────────────────────────────────────────
# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 2048
LR_INITIAL  = 0.05              # Adagrad initial LR
SUBSAMPLE_T = 1e-4             # subsampling threshold

# ── Subsampling: compute discard probability per word ─────────────────────────
# Use total_tokens (raw corpus token count) as the denominator, not the sum of
# in-vocab frequencies.  Rare words below MIN_FREQ were collapsed to <UNK> but
# they still existed in the corpus; counting only kept-vocab tokens overstates
# f(w) for frequent words, discarding them more aggressively than intended.
total_count = total_tokens

discard_prob = {}
for i in range(VOCAB_SIZE):
    f = word_freq.get(idx_to_word[i], 0) / total_count
    discard_prob[i] = max(0.0, 1.0 - math.sqrt(SUBSAMPLE_T / f)) if f > 0 else 0.0

# ── PyTorch IterableDataset — streams pairs, never loads all into RAM ─────────
class SkipGramIterableDataset(IterableDataset):
    """
    Generates (target, context) skip-gram pairs on-the-fly from the
    encoded corpus with inline subsampling.  No full pair list is ever
    materialised in memory.  Supports multi-worker DataLoader by splitting
    the sentence list evenly across workers.

    Pairs are accumulated into a pre-allocated numpy buffer and yielded
    as whole-batch numpy arrays (shape: (buf_size,)).  This eliminates
    the per-pair torch.tensor() overhead and DataLoader collation cost.
    """

    def __init__(
        self,
        encoded_corpus: list[list[int]],
        window: int,
        discard_prob: dict[int, float],
        buf_size: int = BATCH_SIZE,
    ):
        self.encoded_corpus = encoded_corpus
        self.window         = window
        self.discard_prob   = discard_prob
        self.buf_size       = buf_size

    def _generate(self, sentences: list[list[int]]):
        dp  = self.discard_prob
        w   = self.window
        bs  = self.buf_size
        buf_t = np.empty(bs, dtype=np.int32)
        buf_c = np.empty(bs, dtype=np.int32)
        ptr = 0
        for sentence in sentences:
            n = len(sentence)
            for i, target in enumerate(sentence):
                if random.random() < dp.get(target, 0.0):
                    continue
                lo = max(0, i - w)
                hi = min(n - 1, i + w)
                for j in range(lo, hi + 1):
                    if j == i:
                        continue
                    ctx = sentence[j]
                    if random.random() < dp.get(ctx, 0.0):
                        continue
                    buf_t[ptr] = target
                    buf_c[ptr] = ctx
                    ptr += 1
                    if ptr == bs:
                        yield buf_t.copy(), buf_c.copy()
                        ptr = 0
        if ptr > 0:  # flush final partial buffer
            yield buf_t[:ptr].copy(), buf_c[:ptr].copy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        sentences = self.encoded_corpus
        if worker_info is not None:
            per_worker = math.ceil(len(sentences) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end   = min(start + per_worker, len(sentences))
            sentences = sentences[start:end]
        yield from self._generate(sentences)

# ── Optimiser ─────────────────────────────────────────────────────────────────
# Adagrad adapts the learning rate per parameter: high-frequency words
# accumulate large squared-gradient sums and get smaller effective LRs,
# while rare semantic words keep a higher effective LR.  This prevents the
# "frequency collapse" (all embeddings converging to one dominant direction)
# that plague SGD with large batches on this task.
optimizer = optim.Adagrad(model.parameters(), lr=LR_INITIAL)

# ── Training loop ─────────────────────────────────────────────────────────────
epoch_losses: list[float] = []

epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit=" epoch")
for epoch in epoch_bar:
    # Shuffle sentence order each epoch for training variety
    shuffled_corpus = encoded_corpus.copy()
    random.shuffle(shuffled_corpus)

    dataset = SkipGramIterableDataset(shuffled_corpus, WINDOW_SIZE, discard_prob)
    loader  = DataLoader(
        dataset,
        batch_size=None,              # dataset yields pre-batched numpy arrays
        collate_fn=lambda x: x,       # bypass default collate — it converts numpy→tensor
        num_workers=4,
        prefetch_factor=4,            # keep 4 batches queued per worker
        persistent_workers=True,      # avoid worker respawn each epoch
    )

    running_loss = 0.0
    step         = 0

    batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit=" batch", leave=False)
    for step, (target_ids, pos_ctx_ids) in enumerate(batch_bar):
        # Adagrad handles LR adaptation internally — no manual schedule needed

        # Workers yield numpy arrays — one from_numpy call per batch, not per pair
        B           = len(target_ids)
        target_ids  = torch.from_numpy(target_ids).long().to(DEVICE)
        pos_ctx_ids = torch.from_numpy(pos_ctx_ids).long().to(DEVICE)
        # Sample negatives directly on GPU
        neg_ids = sample_negatives(B, NUM_NEGATIVES)  # already on DEVICE

        optimizer.zero_grad()
        loss = model(target_ids, pos_ctx_ids, neg_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        batch_bar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

    epoch_avg = running_loss / max(step + 1, 1)
    epoch_losses.append(epoch_avg)
    epoch_bar.set_postfix(loss=f"{epoch_avg:.4f}")
    tqdm.write(f"✓ Epoch {epoch}/{EPOCHS} complete — avg loss: {epoch_avg:.4f}")

print("\nTraining complete.")


# ── Plot training loss curve ──────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Average Negative-Sampling Loss")
plt.title("Word2Vec Skip-gram Training Loss")
plt.xticks(range(1, EPOCHS + 1))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("model_artifacts")
SAVE_DIR.mkdir(exist_ok=True)

# Extract trained embedding matrices
# Both W_in (target) and W_out (context) are saved.  Averaging the two
# consistently improves retrieval quality (Pennington et al.; Levy et al., 2015)
# because W_out encodes complementary co-occurrence signal.

embeddings_in:  np.ndarray = model.W_in.weight.detach().cpu().numpy()   # (V, 300)
embeddings_out: np.ndarray = model.W_out.weight.detach().cpu().numpy()  # (V, 300)
embeddings: np.ndarray = (embeddings_in + embeddings_out) / 2.0         # averaged
np.save(SAVE_DIR / "embeddings.npy",     embeddings)
np.save(SAVE_DIR / "embeddings_in.npy",  embeddings_in)
np.save(SAVE_DIR / "embeddings_out.npy", embeddings_out)

# Save vocab mappings
with open(SAVE_DIR / "word_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(word_to_idx, f, ensure_ascii=False)

with open(SAVE_DIR / "idx_to_word.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx_to_word.items()}, f, ensure_ascii=False)

print(f"Saved to {SAVE_DIR.resolve()}")
print(f"  embeddings.npy     — averaged (W_in+W_out)/2, shape {embeddings.shape}")
print(f"  embeddings_in.npy  — W_in only, shape {embeddings_in.shape}")
print(f"  embeddings_out.npy — W_out only, shape {embeddings_out.shape}")
print(f"  word_to_idx.json — {len(word_to_idx):,} entries")
print(f"  idx_to_word.json — {len(idx_to_word):,} entries")


# ─────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import json

# ── Reload saved artefacts ────────────────────────────────────────────────────
SAVE_DIR = Path("model_artifacts")

embeddings  = np.load(SAVE_DIR / "embeddings.npy")         # (V, 300)
with open(SAVE_DIR / "word_to_idx.json", encoding="utf-8") as f:
    word_to_idx = json.load(f)
with open(SAVE_DIR / "idx_to_word.json", encoding="utf-8") as f:
    idx_to_word = {int(k): v for k, v in json.load(f).items()}

VOCAB_SIZE, EMBED_DIM = embeddings.shape
print(f"Reloaded embeddings: {embeddings.shape}")

# ── All-but-the-Top post-processing (Mu et al., 2018) ────────────────────────
# Trained word embeddings are highly anisotropic: all vectors cluster in a
# narrow cone of the embedding space, making cosine similarities spuriously
# high between unrelated words.  The fix:
#   1. Subtract the mean embedding (centres the distribution at the origin).
#   2. Remove the top-D principal components (which capture the dominant shared
#      direction that is common to all word vectors and carries no word-specific
#      semantic information).
# This is applied once as post-processing; no retraining required.

D_REMOVE = 1   # number of dominant PCs to remove

def all_but_the_top(vecs: np.ndarray, d: int) -> np.ndarray:
    """
    Mean-centre and remove the top-d principal components from embedding matrix.
    Args:
        vecs: (V, E) float32 embedding matrix
        d:    number of PCs to remove
    Returns:
        (V, E) post-processed matrix (float32)
    """
    # Step 1: mean-centre
    mean_vec = vecs.mean(axis=0, keepdims=True)   # (1, E)
    vecs_c   = vecs - mean_vec                    # (V, E)
    # Step 2: PCA via SVD on a sample (full SVD on 50k×300 is fast enough)
    _, _, Vt = np.linalg.svd(vecs_c, full_matrices=False)  # Vt: (E, E)
    top_components = Vt[:d]                        # (d, E) — top-d PCs
    # Step 3: project out the top-d PCs from every vector
    projection = vecs_c @ top_components.T @ top_components  # (V, E)
    return (vecs_c - projection).astype(np.float32)

print(f"Applying All-but-the-Top (removing {D_REMOVE} dominant PCs) ...")
embeddings = all_but_the_top(embeddings, D_REMOVE)
print("Post-processing complete.")

# ── L2-normalise all vectors once (speeds up cosine search) ──────────────────
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0                # avoid division by zero
unit_embeddings = embeddings / norms   # (V, 300), unit vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vector(word: str) -> np.ndarray | None:
    """Return the embedding vector for a word, or None if OOV."""
    idx = word_to_idx.get(word.lower())
    return embeddings[idx] if idx is not None else None


def most_similar(word: str, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Return top_n most similar words to `word` by cosine similarity.
    Uses pre-normalised matrix for efficiency.
    """
    idx = word_to_idx.get(word.lower())
    if idx is None:
        return []
    query_vec = unit_embeddings[idx]                     # (300,)
    scores    = unit_embeddings @ query_vec              # (V,)  dot = cosine (unit vecs)
    scores[idx] = -1.0                                   # exclude the word itself
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(idx_to_word[i], float(scores[i])) for i in top_indices]

print("Helper functions ready.")


# ─────────────────────────────────────────────────────────────────────────
QUERY_WORDS = ["coffee", "pasta", "tuna", "cookies"]
TOP_N       = 10

for word in QUERY_WORDS:
    results = most_similar(word, top_n=TOP_N)
    if not results:
        print(f"'{word}' not in vocabulary.\n")
        continue
    print(f"{'─'*45}")
    print(f"  Top {TOP_N} words most similar to  '{word}'")
    print(f"{'─'*45}")
    for rank, (neighbor, score) in enumerate(results, start=1):
        print(f"  {rank:2d}.  {neighbor:<20s}  cosine = {score:.4f}")
    print()


# #### Discussion — Word Similarity Results
# 
# **coffee**: Likely neighbours include *tea*, *cocoa*, *espresso*, *beverage*. Wikipedia discusses coffee in agricultural, trade, cultural, and health contexts, so results tend toward beverage names and origin countries rather than "morning cup" associations.
# 
# **pasta**: Expect *spaghetti*, *risotto*, *sauce*, *Italian*, *noodles*. Wikipedia emphasises cuisine, Italian culture, and ingredient relationships.
# 
# **tuna**: Likely *bluefin*, *swordfish*, *salmon*, *species*, *yellowfin*. Wikipedia's coverage is taxonomic and marine-biology-heavy, so semantically close words name fish species and fishing methods.
# 
# **cookies**: Web-browsing cookies and baked cookies share the token form "cookies" — results may blend both senses (e.g., *biscuit* vs. *browser*, *session*). This **polysemy** is a known limitation of single-sense static embeddings like Word2Vec.
# 
# **General observation**: WikiText neighbours tend toward factual, encyclopedic co-occurrences rather than conversational or culinary associations. A corpus like Common Crawl or recipe blogs would yield richer food-domain similarity.

# ─────────────────────────────────────────────────────────────────────────
import numpy as np
from pathlib import Path

GLOVE_PATH = Path("glove.6B.300d.txt")   # ← update path if needed

def load_glove(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """
    Parse a GloVe text file.
    Returns:
        glove_dict  — word → np.ndarray (300,)
        glove_matrix — (V_glove, 300) matrix for fast batch search
        glove_words  — list of words in matrix row order
    """
    glove_dict: dict[str, np.ndarray] = {}
    print(f"Loading GloVe from {path} ...")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vec

    glove_words  = list(glove_dict.keys())
    glove_matrix = np.stack([glove_dict[w] for w in glove_words])  # (V, 300)
    # L2-normalise
    norms = np.linalg.norm(glove_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    glove_matrix_unit = glove_matrix / norms

    print(f"GloVe loaded: {len(glove_words):,} words, dim={glove_matrix.shape[1]}")
    return glove_dict, glove_matrix_unit, glove_words


if not GLOVE_PATH.exists():
    print(f"[WARNING] GloVe file not found at '{GLOVE_PATH}'.")
    print("  Please download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip")
    print("  and extract glove.6B.300d.txt to the working directory.")
    glove_dict, glove_matrix_unit, glove_words = {}, np.zeros((1, 300)), []
else:
    glove_dict, glove_matrix_unit, glove_words = load_glove(GLOVE_PATH)

glove_word_to_idx = {w: i for i, w in enumerate(glove_words)}


# ─────────────────────────────────────────────────────────────────────────
def solve_analogy(
    a: str, b: str, c: str,
    glove: dict[str, np.ndarray],
    glove_unit: np.ndarray,
    word_to_idx: dict[str, int],
    idx_to_word: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Solve  A : B :: C : ?
    Returns top_n (word, cosine_similarity) candidates.
    Excludes a, b, c from results.
    """
    for w in [a, b, c]:
        if w not in glove:
            print(f"  [WARN] '{w}' not in GloVe vocabulary — skipping.")
            return []

    # Raw target vector (not normalised yet)
    v_target = glove[b] - glove[a] + glove[c]
    v_target_unit = v_target / (np.linalg.norm(v_target) + 1e-9)

    scores = glove_unit @ v_target_unit       # (V,)  cosine similarities

    # Mask out a, b, c
    for w in [a, b, c]:
        if w in word_to_idx:
            scores[word_to_idx[w]] = -1.0

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(idx_to_word[i], float(scores[i])) for i in top_idx]


# ── Define the four analogies ─────────────────────────────────────────────────
analogies = [
    ("spain",     "spanish",  "germany",   "Expected: german    [country → language]"),
    ("japan",     "tokyo",    "france",    "Expected: paris     [country → capital]"),
    ("woman",     "man",      "queen",     "Expected: king      [gender / role]"),
    ("australia", "hotdog",   "italy",     "Expected: pizza/pasta? [country → food — weak relation]"),
]

if not glove_words:
    print("[INFO] GloVe not loaded — skipping analogy evaluation.")
else:
    for a, b, c, label in analogies:
        print(f"\n{'═'*55}")
        print(f"  {a.upper()} : {b.upper()} :: {c.upper()} : ?")
        print(f"  {label}")
        print(f"{'─'*55}")
        results = solve_analogy(
            a, b, c,
            glove_dict, glove_matrix_unit, glove_word_to_idx, glove_words, top_n=5
        )
        for rank, (word, score) in enumerate(results, start=1):
            print(f"  {rank}. {word:<20s}  cosine = {score:.4f}")


# ─────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ── Select top-200 frequent words (skip <UNK>) ────────────────────────────────
# Use the word_freq counter from Part 1 (re-import if running standalone)
try:
    freq_items = [(w, word_freq[w]) for w in word_to_idx
                  if w != "<UNK>" and w in word_freq]
except NameError:
    # If running standalone, rebuild from saved vocab (no frequency info)
    freq_items = [(idx_to_word[i], 1) for i in range(1, min(201, VOCAB_SIZE))]

top200_words = [w for w, _ in sorted(freq_items, key=lambda x: -x[1])[:200]]
top200_idx   = [word_to_idx[w] for w in top200_words]
top200_vecs  = embeddings[top200_idx]    # (200, 300)

# ── Run t-SNE ─────────────────────────────────────────────────────────────────
print("Running t-SNE (this takes ~30 s) ...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
            random_state=42, init="pca", learning_rate="auto")
vecs_2d = tsne.fit_transform(top200_vecs)  # (200, 2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 14))
ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=12, alpha=0.7)

for i, word in enumerate(top200_words):
    ax.annotate(word, (vecs_2d[i, 0], vecs_2d[i, 1]),
                fontsize=7, alpha=0.85,
                xytext=(2, 2), textcoords="offset points")

ax.set_title("t-SNE of Top-200 Word Embeddings (Word2Vec Skip-gram)", fontsize=14)
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
print("t-SNE plot complete. Look for semantic clusters (e.g., numbers, pronouns, proper nouns).")


 
# ### Files produced
# - `model_artifacts/embeddings.npy` — trained embedding matrix
# - `model_artifacts/word_to_idx.json` — word → index
# - `model_artifacts/idx_to_word.json` — index → word
