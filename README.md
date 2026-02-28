# Word2Vec Skip-Gram with Negative Sampling

PyTorch implementation of Skip-gram Word2Vec trained on WikiText-103. Assignment 1 for CS 797Y NLP at Wichita State University.

## Features

- Skip-gram architecture with negative sampling
- Memory-efficient streaming dataset (handles ~800M pairs)
- All-but-the-Top post-processing for better embeddings
- Word similarity and analogy evaluation

## Results

- **Dataset**: WikiText-103 (80.4M tokens, 47.7K vocabulary)
- **Model**: 300D embeddings, 10 epochs, final loss 2.61
- **Word similarity**: coffee → cocoa, tea, beans, drinks
- **Analogies**: Spain:Spanish :: Germany:german (0.90)

## Quick Start

```bash
git clone https://github.com/alisenby94/word2vec-skipgram.git
cd word2vec-skipgram
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download GloVe for analogies
wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip

# Run (uses included pre-trained artifacts, or retrain from scratch)
python word2vec_assignment.py
```

Training from scratch takes ~4-6 hours on GPU. Pre-trained artifacts (166MB) are included.

## Project Structure

```
├── word2vec_assignment.py    # Main script (679 lines)
├── model_artifacts/           # Pre-trained embeddings (166MB, included)
├── report/word2vec_report.pdf # Technical report
├── requirements.txt
└── README.md
```

## Implementation

**Architecture**: Two 300D embedding matrices (input/output), negative sampling loss with 10 negatives

**Training**: Adagrad optimizer, batch size 2048, window size 5, min word frequency 50

**Key features**: Streaming IterableDataset, GPU-accelerated negative sampling, frequency-based subsampling, All-but-the-Top post-processing

See [`report/word2vec_report.pdf`](report/word2vec_report.pdf) for full details.

## References

Mikolov et al. (2013), Pennington et al. (2014), Merity et al. (2016), Mu & Viswanath (2018)

## License

MIT License

---

**Andrew Lisenby** | Wichita State University | [GitHub](https://github.com/alisenby94)
