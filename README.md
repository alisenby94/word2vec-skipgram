# Word2Vec Skip-Gram Implementation with Negative Sampling

A from-scratch PyTorch implementation of the Skip-gram Word2Vec model with negative sampling, trained on the WikiText-103 dataset. This project was completed as Assignment 1 for CS 797Y Natural Language Processing at Wichita State University.

## Overview

This project implements the Skip-gram Word2Vec architecture as described by Mikolov et al. (2013), featuring:

- **Clean Python implementation** using PyTorch with two separate embedding tables (input and output)
- **Negative sampling loss** for efficient training without full softmax computation
- **Streaming dataset** using IterableDataset to handle ~800M training pairs without memory overflow
- **All-but-the-Top post-processing** to reduce embedding anisotropy and improve cosine similarity reliability
- **Comprehensive evaluation** including word similarity neighborhoods and vector arithmetic analogies with GloVe

### Key Results

- **Dataset**: WikiText-103 (80.4M tokens, 47,688-word vocabulary)
- **Embedding Dimension**: 300D
- **Training epochs**: 10
- **Final loss**: 2.6132
- **Optimizer**: Adagrad (initial LR: 0.05) with batch size 2048

#### Word Similarity Examples
The model learns semantically coherent neighborhoods:
- **coffee** → cocoa, tea, beans, drinks, vodka, etc.
- **pasta** → soups, salad, dishes, desserts, sauces, etc.
- **tuna** → mackerel, shrimp, squid, lobsters, fish, etc.
- **cookies** → biscuits, desserts, snacks, pancakes, etc.

#### Word Analogies (GloVe 300d)
Vector arithmetic validates semantic/syntactic capture:
- Spain : Spanish :: Germany : **german** (0.8975)
- Japan : Tokyo :: France : **paris** (0.8097)
- Woman : Man :: Queen : **king** (0.6635)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/alisenby94/word2vec-skipgram.git
cd word2vec-skipgram
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download GloVe (required for Part 3)
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Run complete pipeline (training + evaluation)
python word2vec_assignment.py

# Or skip training and use pre-trained artifacts (Parts 2-3 only)
# Comment out Part 1 code in the script, or modify to load existing artifacts
```

**Note**: 
- Training takes ~4-6 hours on GPU or ~24+ hours on CPU
- Pre-trained model artifacts (166MB) are included in the repository
- You can evaluate immediately without retraining

## Setup

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alisenby94/word2vec-skipgram.git
cd word2vec-skipgram
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download GloVe embeddings (for analogy evaluation):
```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## Usage

### Running the Complete Pipeline

Run the main script to execute all three parts:
```bash
python word2vec_assignment.py
```

The script will:
1. **Part 1: Training**
   - Download and preprocess the WikiText-103 dataset (103M tokens → 80.4M after cleaning)
   - Build vocabulary with minimum frequency threshold of 50 (→ 47,688 words)
   - Train Skip-gram model with negative sampling for 10 epochs (~4-6 hours on GPU)
   - Display training loss curves
   - Save model artifacts to `model_artifacts/`
   
   **Note**: You can skip Part 1 if using the pre-trained artifacts included in the repository.

2. **Part 2: Word Similarity**
   - Apply All-but-the-Top post-processing to embeddings
   - Find top-10 most similar words for query words (coffee, pasta, tuna, cookies)
   - Display cosine similarity scores

3. **Part 3: Word Analogies**
   - Load pre-trained GloVe 300d embeddings
   - Solve analogies using vector arithmetic (A : B :: C : ?)
   - Evaluate on country-language, country-capital, and gender analogies

4. **Bonus: Visualization**
   - Generate t-SNE 2D projection of top-200 most frequent words
   - Display semantic clustering

### Model Artifacts

After training, the following files are saved in `model_artifacts/`:
- `embeddings_in.npy` - Input (target) embedding matrix [47,688 × 300]
- `embeddings_out.npy` - Output (context) embedding matrix [47,688 × 300]
- `embeddings.npy` - Averaged embeddings after All-but-the-Top post-processing
- `word_to_idx.json` - Word to index mapping (47,688 entries)
- `idx_to_word.json` - Index to word mapping (47,688 entries)

**Note**: Pre-trained model artifacts are included in the repository (166MB) for convenience. You can use them directly or retrain the model to generate new ones.

## Project Structure

```
.
├── word2vec_assignment.py     # Main implementation (679 lines, cleaned from notebook)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # This file
├── .gitignore                # Git ignore rules
│
├── report/
│   ├── word2vec_report.pdf   # Final compiled report (included in repo)
│   └── ...                   # LaTeX sources (excluded from repo)
│
├── model_artifacts/          # Pre-trained artifacts (included in repo)
│   ├── embeddings.npy        # Averaged embeddings (post-processed) [55MB]
│   ├── embeddings_in.npy     # Input embeddings [55MB]
│   ├── embeddings_out.npy    # Output embeddings [55MB]
│   ├── word_to_idx.json      # Vocabulary mapping [842KB]
│   └── idx_to_word.json      # Reverse vocabulary mapping [935KB]
│
└── glove.6B.*.txt            # GloVe embeddings (excluded from repo, download manually)
```

### What's Included in the Repository

**Included:**
- ✅ Main implementation script (`word2vec_assignment.py`)
- ✅ Dependencies list (`requirements.txt`)
- ✅ Final report PDF (`report/word2vec_report.pdf`)
- ✅ Pre-trained model artifacts (`model_artifacts/`, 166MB total)
- ✅ License and documentation

**Excluded (too large or need local setup):**
- ❌ GloVe embeddings (~1GB) - download manually
- ❌ LaTeX source files - only PDF included
- ❌ Virtual environment - create locally
- ❌ Archive directory - old notebook versions

## Implementation Details

### Architecture

The Skip-gram model consists of two embedding matrices:
- **Input embeddings** (W_in): 47,688 × 300
- **Output embeddings** (W_out): 47,688 × 300

### Training Objective

Negative sampling loss:

$$\mathcal{L} = -\log \sigma(v_c^\top u_o) - \sum_{k=1}^{10} \log \sigma(-v_c^\top u_k)$$

Where:
- $v_c$ = target word vector
- $u_o$ = true context vector
- $u_k$ = 10 negative samples
- $\sigma$ = sigmoid function

### Training Strategy

- **Dataset**: WikiText-103 with on-the-fly skip-gram pair generation using IterableDataset
- **Context window**: 5 (captures medium-range syntactic and topical co-occurrences)
- **Minimum word frequency**: 50 (filters rare words to reduce noise)
- **Batch size**: 2048 (reduced from 8192 for better convergence)
- **Optimizer**: Adagrad with initial learning rate 0.05
- **Subsampling**: Frequent words discarded with probability based on frequency
- **Multi-worker DataLoader**: 4 workers with prefetching for efficient CPU→GPU pipeline
- **Gradient clipping**: Max norm 1.0 to prevent exploding gradients
- **Post-processing**: All-but-the-Top (removes 1 dominant principal component)

### Key Implementation Features

1. **Memory-Efficient Streaming**: Uses `IterableDataset` to generate skip-gram pairs on-the-fly, avoiding the need to materialize ~800M training pairs in RAM (which would require >50GB)

2. **GPU-Accelerated Negative Sampling**: Samples negatives directly on GPU using `torch.multinomial`, eliminating CPU→GPU transfer overhead

3. **Frequency-Based Subsampling**: Implements Mikolov's subsampling to reduce frequent words (e.g., "the", "of"), improving training on rare semantic words

4. **All-but-the-Top Post-Processing**: Applies Mu et al. (2018) technique to reduce anisotropy and improve cosine similarity reliability

## Results

### Training Performance

- **Final loss**: 2.6132 (converged from 2.81 in epoch 1)
- **Training time**: ~4-6 hours on NVIDIA GPU
- **Vocabulary coverage**: 47,688 words (from 530K+ raw tokens)

### Evaluation Results

See `report/word2vec_report.pdf` for complete analysis including:
- **Training loss curves** showing convergence over 10 epochs
- **t-SNE visualization** of the top-200 most frequent word embeddings
- **Word similarity tables** with cosine similarity scores
- **Analogy evaluation** using pre-trained GloVe embeddings

See `report/word2vec_report.pdf` for complete analysis including:
- Training loss curves showing smooth convergence
- t-SNE visualization of top-200 most frequent word embeddings
- Detailed word similarity tables with cosine scores
- Word analogy evaluation results
- Discussion of model behavior on encyclopedic (Wikipedia) text

## Technical Report

The complete technical report is available at [`report/word2vec_report.pdf`](report/word2vec_report.pdf) and includes:

- **Abstract**: Summary of implementation and key results
- **Introduction**: Background on Word2Vec and project objectives
- **Preprocessing**: Tokenization strategy and vocabulary construction
- **Model Architecture**: Detailed explanation of Skip-gram with negative sampling
- **Training Process**: Hyperparameter choices, optimization strategy, and convergence analysis
- **Experiments**: Word similarity evaluation and analogy results
- **Discussion**: Analysis of semantic neighborhoods on Wikipedia text
- **Conclusion**: Summary and potential improvements

The report follows IEEE conference paper format and includes mathematical formulations, training curves, and qualitative examples.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: WikiText-103 from Hugging Face `datasets` library
- **GloVe**: Pre-trained embeddings by Pennington et al. (Stanford NLP)
- **Course**: CS 797Y Natural Language Processing, Wichita State University

## References

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532-1543).
- Mu, J., & Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. *In International Conference on Learning Representations*.

## Author

**Andrew Lisenby**  
School of Computing, Wichita State University  
Email: ablisenby@shockers.wichita.edu  
GitHub: [@alisenby94](https://github.com/alisenby94)

---

*Completed as Assignment 1 for CS 797Y Natural Language Processing, Spring 2026*
