# Substitution Cipher Solver

Cracking simple substitution ciphers (26! ≈ 4×10²⁶ possible keys — brute force is impossible) by combining a **character n-gram language model** with **local search** (hill climbing / simulated annealing).

## The idea

A substitution cipher replaces every letter with another using a secret key (a permutation of the alphabet). Original text `W` goes through the "channel" (encryption) and comes out as ciphertext `C`. We want:

```
argmax_W  P(W | C)  ∝  P(C | W) · P(W)
```

Encryption is deterministic, so `P(C|W)` is 1 for the key consistent with `C` and 0 otherwise — the whole problem reduces to **finding the key whose decryption has the highest language-model probability**. The LM is the judge; the search algorithm explores keys by swapping letters.

## Pipeline (planned end state)

```
Raw books (NLTK gutenberg corpus) → Cleaning (27-symbol alphabet: a–z + space)
        │
        ├─ 5 train books ──▶ Character n-gram LM (unigram→4-gram, add-k + interpolation)
        └─ 2 test books  ──▶ passages (40 pieces × lengths 50/100/200/500/1000 chars)
                                  │
Plaintext ──▶ encrypt(random key) ──▶ ciphertext
                                  │
        Search: frequency baseline → hill climbing → simulated annealing
        proposes key → decrypt → LM scores it → repeat
                                  │
        best key ──▶ decrypted text ──▶ post-processing (dictionary + edit distance)
                                  │
        Evaluation (experiments + graphs) + Streamlit demo
```


## Project structure

```
cipher-solver/
├── data/
│   ├── raw/       # 7 books as downloaded (untouched originals)
│   ├── clean/     # 5 cleaned TRAIN books (27-symbol alphabet)
│   └── test/      # 2 cleaned TEST books + passages/ (10 files, 40 pieces each)
├── src/
│   ├── data.py    # cleaning, split routing, alphabet verification, passages
│   ├── lm.py      # CharNGram: train / logprob / perplexity   
│   ├── cipher.py  # generate_key, encrypt, decrypt
│   ├── solvers.py # frequency baseline, hill climbing, SA
│   ├── postprocess.py  # dictionary + edit distance fixer
│   ├── evaluate.py     # key/char/word accuracy metrics
│   └── experiments.py  # run the grid, save CSVs
├── models/        # saved LM counts
├── results/       # CSVs + graphs
├── app/           # Streamlit demo
├── tests/         # checkpoint scripts
└── requirements.txt
```

## Data

**Provenance:** Project Gutenberg texts via the NLTK gutenberg corpus, fetched as raw `.txt` files from the `nltk_data` GitHub repo (gutenberg.org was returning 503 at setup time; the GitHub-hosted corpus and the Waterloo mirror were verified reachable).

| Book | Role | Clean chars |
|---|---|---|
| austen-emma | train | 845,320 |
| austen-persuasion | train | 448,511 |
| austen-sense | train | 645,040 |
| chesterton-ball | train | 436,307 |
| chesterton-thursday | train | 307,403 |
| **train total** | | **2,682,581** |
| melville-moby_dick | test — unseen author, same genre | 1,167,934 |
| shakespeare-hamlet | test — genre shift (verse drama) | 151,996 |

- Raw books contain 67–90 distinct symbols (punctuation, digits, control chars, plus `è é î` in chesterton-ball); cleaning reduces everything to exactly 27.
- Cleaning rules: lowercase → remove apostrophes (`'`, backtick, U+2019, U+2018) → every run of non-`a-z` characters becomes one space → strip. (`don't` → `dont`, never `don t`.)
- Test passages: 40 evenly spaced pieces per (book × length), one piece per line in `data/test/passages/<book>_<L>.txt`.
- Deliberate choices: Dickens dropped (5 training books ≈ 2.68M chars is ample for 4-grams over 27 symbols); **all** Shakespeare kept out of training so Hamlet is an unambiguous genre-shift test; the Bible, verse and short texts excluded to keep the LM's register consistent.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Cmder/PowerShell)
pip install -r requirements.txt   # nltk, numpy, pandas, matplotlib (streamlit comes later)
```

## Usage

```bash
python src/data.py    # clean books → data/clean (5) + data/test (2), verify alphabet, build passages
python src/lm.py      # train unigram + bigram char models, print sanity checks   (in progress)
```

## Status

| # | Task | Status |
|---|---|---|
| 0 | Environment + skeleton | ✅ |
| 1 | Books downloaded | ✅ |
| 2 | Cleaning pipeline | ✅ |
| 3 | Test passages | ✅ |
| 4 | Unigram + bigram LM | ✅ |
| 5 | Trigram + 4-gram + interpolation; perplexity table | ✅ |
| 6 | Cipher module | ✅ |
| 7 | Frequency baseline solver | ✅ |
| 8 | Hill climbing + restarts | ✅ |
| 9 | Simulated annealing + tuning | ✅ |
| 10 | Post-processing (dictionary + edit distance) | ✅ |
| 11 | Experiments + graphs | ✅ |
| 12 | Streamlit demo | ✅ |


