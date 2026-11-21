# Substitution Cipher Solver: Project Overview

## 1. Project goal

This project builds an automated solver for a **simple substitution cipher**. In this cipher, every plaintext letter is replaced by a different ciphertext letter, while spaces can remain as a separate visible symbol.

For example, a key might state:

```text
plaintext letter:  a b c d e f g h i j k l m n o p q r s t u v w x y z
ciphertext letter:  q x m e n j r w f a c b i z r k p g v l s h t u d t
```

The key is a permutation: each of the 26 letters appears exactly once on both sides. This is not an encryption system intended for security. The purpose is to study probabilistic language models and search algorithms.

The solver does not try all 26! possible keys. It would be impossible to test approximately 4 × 10²⁶ permutations. Instead, it estimates how much a candidate plaintext resembles English and searches for a key that maximizes that score.

The central problem is therefore:

```text
Choose the key W that maximizes P(W | C),
where W is the decryption candidate and C is the ciphertext.
```

Using Bayes' rule:

```text
P(W | C) ∝ P(C | W) × P(W)
```

Because encryption is deterministic, a key produces exactly one plaintext. Consequently, the search is mainly about maximizing the language probability `P(W)`.

## 2. End-to-end pipeline

The project follows this pipeline:

```text
raw Project Gutenberg books
        ↓
cleaning and alphabet normalization
        ↓
five training books + two unseen test books
        ↓
character n-gram language models
        ↓
random substitution encryption
        ↓
ciphertext passage
        ↓
frequency baseline
        ↓
hill climbing
        ↓
simulated annealing
        ↓
best recovered text
        ↓
post-processing with a dictionary and edit distance
        ↓
accuracy evaluation and Streamlit demonstration
```

The raw books are separate from the cleaned, testing, and model-training outputs. The cleaned corpus is the data from which the solver learns English spelling patterns.

## 3. Corpus construction

### Raw books

The project downloads seven Project Gutenberg-style books:

**Training books**

- `austen-emma.txt`
- `austen-persuasion.txt`
- `austen-sense.txt`
- `chesterton-ball.txt`
- `chesterton-thursday.txt`

**Test books**

- `melville-moby_dick.txt`
- `shakespeare-hamlet.txt`

Moby Dick and Hamlet are intentionally excluded from training.

Moby Dick provides an unseen-author test. Hamlet provides a genre-shift test because it contains dramatic verse and dialogue rather than the same general prose style as the training books.

### Why five training books?

Five books are a practical balance between:

- Corpus size
- Training reliability
- Runtime
- Domain consistency
- No data leakage

The total cleaned training corpus is approximately 2.68 million characters. This is large enough to provide useful counts for up to four-character patterns.

The character alphabet is small: 27 symbols. Therefore, 2.68 million characters is ample for experimenting with unigram through fourgram models.

Five books also keep the training data independent of the two test books. A larger or different training set could improve or change the solver, but adding Moby Dick or Shakespeare would make the held-out test invalid.

## 4. Cleaning and alphabet normalization

The cleaned text is converted to lowercase. Apostrophes and their typographic variants are removed before processing. Every sequence of characters that is not `a-z` is replaced by a space.

The implemented cleaning rules are:

```text
lowercase
remove apostrophes: '  `  ’  ‘
replace every non-letter run with a space
collapse whitespace
strip leading and trailing spaces
```

For example:

```text
"Don't stop." -> "dont stop"
```

This intentionally produces `dont`, not `don t`. The latter would add a fake word boundary and distort character counts.

### Why exactly 27 symbols?

The 26 lowercase English letters are represented, plus space. Space is meaningful because it records word boundaries.

The cipher key still permutes only the 26 letters. Cleaning does not create cipher mappings for punctuation or digits.

The model vocabulary is therefore:

```text
V = 27
```

This lets an n-gram model learn patterns such as:

- `th`
- `he `
- `and`
- ` of`
- `the `

The value `27` is also used for add-k smoothing. It represents the number of possible next characters, including space.

## 5. Keeping or removing spaces

Spaces are initially kept because word boundaries contain useful information. However, the project also tests the space-removed experiment:

**Spaces kept**

```text
"the king"
```

The model learns both letter patterns and word-boundary patterns.

**Spaces removed**

```text
"theking"
```

The solver treats the decrypted passage as one continuous character sequence. This creates a harder substitution problem because word boundaries are no longer directly visible.

Why test both?

- It measures how much accuracy comes from word-boundary information.
- It tests whether character n-grams alone can solve the problem.
- It establishes a planned experiment axis for later performance tables.
- It helps distinguish language-model quality from spacing information.

In spaces-kept mode, the 26-letter substitution key remains unchanged. Only the plaintext representation is changed for a separate experimental condition.

## 6. Character n-gram language models

An n-gram is a sequence of `n` consecutive characters.

Examples:

| Order | Examples |
|---|---|
| 1 | `t`, `h`, `e`, space |
| 2 | `th`, `he`, `e ` |
| 3 | `the`, `he ` |
| 4 | `the ` |

The project trains models with:

- Unigram: n = 1
- Bigram: n = 2
- Trigram: n = 3
- Fourgram: n = 4

### Training

For a text and an n-gram model, the code counts:

```text
ngram_count[g] = number of times g appears
context_count[c] = number of times context c appears
```

For a bigram `"th"`:

```text
P(h | t) = count("th") / count("t")
```

For a trigram `"the"`:

```text
P(e | th) = count("the") / count("th")
```

### Add-k smoothing

Plain counts can assign probability zero to an unseen pattern:

```text
P(g) = 0
```

Taking `log(0)` gives negative infinity. A cipher solver with several unseen patterns can therefore assign `-inf` to all candidates and become unable to rank them.

Add-k smoothing solves this by pretending every n-gram has received a small additional count:

```text
P(g | context) =
    (count(g) + k) /
    (count(context) + k × 27)
```

The project uses `k = 0.01` as an initial smoothing value and also tests several smoothing values in the perplexity grid.

Smoothing does not remove all unseen patterns. It makes rare and unseen patterns possible and improbable rather than impossible.

### Why log probabilities?

The probability of a long text is the product of many individual probabilities. This product becomes extremely small:

```text
0.03 × 0.03 × 0.03 × ...
```

The value can underflow to zero in a computer. Logs convert multiplication into addition:

```text
log(P1 × P2 × P3) = log(P1) + log(P2) + log(P3)
```

Therefore, the LM returns a summed log score. Larger, or less negative, is better.

Perplexity is used to compare texts of different lengths:

```text
perplexity = exp(-logprob / number_of_events)
```

Lower perplexity means that the text looks more like the language used in training.

### Interpolation

The project also interpolates unigram, bigram, trigram, and fourgram probabilities:

```text
P_interpolated =
    w1 × P1 +
    w2 × P2 +
    w3 × P3 +
    w4 × P4
```

Interpolation combines general information from lower n-grams with specific context from higher n-grams. It prevents a high n-gram model from depending entirely on rare patterns.

The available weights are renormalized where an n-gram is not yet available at the beginning of a string.

## 7. Cipher generation and encryption

`cipher.py` implements:

- Random generation of a key permutation
- Encryption of plaintext
- Decryption of ciphertext

The key is stored as:

```python
{"a": "q", "b": "x", ...}
```

Encryption maps plaintext letters through this key.

```text
plaintext -> cipher key -> ciphertext
```

Decryption reverses the mapping:

```text
ciphertext -> inverse key -> plaintext
```

A round-trip test verifies:

```text
decrypt(encrypt(text, key), key) == text
```

Spaces and characters outside `a-z` are preserved in this module.

## 8. Why not brute force?

An alphabet of 26 letters has:

```text
26! ≈ 4 × 10²⁶
```

possible substitution keys. Exhaustive search is infeasible.

The solver instead performs heuristic search. It starts with a candidate key and repeatedly changes that key. The LM judges the resulting plaintext.

The substitution constraint is maintained: every swap exchanges plaintext letters for cipher letters, so the candidate remains a valid permutation.

## 9. Frequency baseline

The first solver is a simple frequency baseline.

It ranks ciphertext letters by occurrence and ranks English letters by typical frequency:

```text
etaoinshrdlcumwfgypbvkjxqz
```

The most frequent ciphertext letter is mapped to `e`, the next to `t`, and so on.

### Why use it?

- It is easy to explain and implement.
- It provides a fast, deterministic starting point.
- It establishes a weak baseline.
- It gives local-search algorithms a useful starting key.

### Why can it fail?

Letter-frequency order is only a statistical expectation. Small passages do not contain enough examples of all letters. Genre, topic, punctuation, and the actual passage can differ substantially from general English frequency.

This is why the project tests 50-, 100-, 200-, 500-, and 1000-character passages. It is expected to improve somewhat with longer text, but it is not a general solution.

## 10. Hill climbing

Hill climbing starts from a candidate key, typically the frequency baseline key or a random restart.

At each step:

1. Swap two letters in the key.
2. Decrypt the ciphertext.
3. Score the recovered plaintext with the LM.
4. Keep the swap only if the score improves.
5. Stop when no beneficial swap is found.

The algorithm:

```text
while a better swap exists:
    try a random two-letter swap
    keep it only if LM score increases
```

### Why hill climbing?

- It is simple and conceptually clear.
- It uses the LM score directly.
- It is faster than methods that accept every move.
- It is a natural first local-search technique.

### Why it can fail

Hill climbing accepts only improvements, so it can become trapped in a local maximum. Two-letter swaps have a strong idea of order: a key may look good according to the LM but require several simultaneous swaps before it improves.

Multiple random restarts reduce this risk:

```text
run several independent searches
keep the best result across all runs
```

The first restart can start from the frequency baseline. Later restarts can start from random valid keys.

## 11. Simulated annealing

Simulated annealing is a search algorithm inspired by thermal annealing.

It proposes the same random two-letter key swaps as hill climbing. A candidate is evaluated with the LM:

```text
delta = candidate_score - current_score
```

An improvement is always accepted.

A worse candidate is accepted with probability:

```text
probability = exp(delta / temperature)
```

The temperature decreases geometrically:

```text
temperature = temperature × cooling
```

A configured floor prevents the temperature from reaching zero.

### Why use simulated annealing?

- It can escape local maxima.
- It explores worse-looking keys early in the search.
- It explores candidate keys near the current key as it cools.
- It becomes more conservative as the temperature decreases.
- It often performs better on a large, rugged search space than plain hill climbing.

### Why choose temperature and cooling?

Temperature controls how tolerant the search is of a score decrease.

- High temperature: more worse moves are accepted.
- Low temperature: worse moves are rarely accepted.
- Slow cooling: more exploration.
- Fast cooling: quicker convergence toward a nearby maximum.

The project tests temperature and cooling combinations on validation text. This is necessary because LM log scores can have large negative values and because the scale of `delta` changes across n-gram orders and text lengths.

## 12. Why these algorithms instead of alternatives?

There are many possible optimization methods:

- Genetic algorithms
- Random search
- Beam search
- Maximum entropy methods
- Dynamic programming
- Neural language models
- Exhaustive key search

The chosen methods are appropriate for this project because:

- **Frequency analysis** is simple and provides a recognized baseline.
- **Hill climbing** is easy to implement and is a direct local-search extension.
- **Simulated annealing** demonstrates how search can escape local maxima.
- **Character n-grams** remain fast enough for millions of key-score evaluations.
- **No neural model is required**, which keeps the project explainable and reproducible.

The project tests increasingly capable search methods rather than choosing a complex algorithm immediately.

## 13. Test passages and accuracy

The test books are divided into passage files:

```text
melville-moby_dick_50.txt
melville-moby_dick_100.txt
melville-moby_dick_200.txt
melville-moby_dick_500.txt
melville-moby_dick_1000.txt

shakespeare-hamlet_50.txt
shakespeare-hamlet_100.txt
shakespeare-hamlet_200.txt
shakespeare-hamlet_500.txt
shakespeare-hamlet_1000.txt
```

Each passage file contains 40 evenly spaced pieces.

The passage lengths are not adjusted to end on a word boundary. This makes the benchmark consistent but intentionally realistic.

## 14. Evaluation metrics

### Key accuracy

Key accuracy measures whether each plaintext-letter mapping was recovered:

```text
key_accuracy =
    number_of_correct_letter_mappings / 26
```

This is strict. Swapping or misidentifying one letter can affect multiple output positions.

### Character accuracy

Character accuracy compares the plaintext and recovered text position by position:

```text
character_accuracy =
    number_of_matching_characters / number_of_characters
```

It includes spaces and letters.

### Word accuracy

Word accuracy compares complete words:

```text
word_accuracy =
    number_of_matching_words / number_of_expected_words
```

One incorrect letter can make an entire word wrong even if most characters are correct.

### Runtime

Runtime is measured using a monotonic process timer. It shows the practical cost of a solver and allows the project to compare the frequency, hill-climbing, and annealing methods.

The three metrics answer different questions:

- Key accuracy: Did the solver find the right permutation?
- Character accuracy: Did the recovered text look close?
- Word accuracy: Did it recover readable words?

## 15. Experiment grid

`experiments.py` combines:

- Passage lengths: 50, 100, 200, 500, 1000
- Solvers: frequency, hill climbing, simulated annealing
- N-gram orders: 1, 2, 3, 4
- Space modes: kept or removed
- Multiple random trials

The script writes:

- `results/experiments.csv`
- `results/summary.csv`
- Six accuracy-versus-length plot images

The grid allows the project to answer questions such as:

- Does more text improve accuracy?
- Does a fourgram outperform a unigram?
- Do spaces help?
- Does annealing outperform hill climbing?
- How much runtime is required?

## 16. Post-processing

The language model searches for a globally plausible decryption. It does not guarantee that every individual word is a real dictionary word.

Post-processing in `postprocess.py` is a second repair stage.

For each word, the module:

1. Lowercases the word.
2. Checks whether it is in a small known-word dictionary.
3. If unknown, computes Levenshtein edit distance to candidate words.
4. Keeps candidates within a maximum edit distance.
5. Ranks candidates by:
   - Edit distance
   - Dictionary frequency
   - Similar character sequence
   - Alphabetical spelling

### Why edit distance?

A single substitution-key error can corrupt every occurrence of certain letters. It can turn an intended word into a near miss, for example:

```text
breakfast -> mreakfast
family -> favily
```

Minimum edit distance measures how many insertions, deletions, or substitutions are needed to transform one word into another. It is useful for repairing a small number of character errors.

### Limitations

The current dictionary is intentionally small. It is sufficient for a checkpoint and common demonstration words but not suitable for unrestricted language.

Post-processing should not replace the cipher key or LM. It is an optional repair stage applied after search.

## 17. Streamlit demo

The Streamlit app provides an interactive demonstration:

- Plaintext input
- N-gram selection
- Solver selection
- Random key generation
- Ciphertext display
- Recovered plaintext display
- Spinner feedback while search runs
- Solver log score and runtime for local search

The app cleans input in the same way as the corpus: lowercase, non-letter sequences become spaces, and whitespace is normalized.

The demo should not be interpreted as proof that a solver always recovers the exact text. It is a controlled demonstration of search, not a guaranteed decoder.

## 18. Interpreting successful results

A close result, such as:

```text
the king of england went to the castle
and then took a rest there until sunrise
```

is evidence that many substitution mappings were recovered.

Residual errors such as:

```text
mreakfast -> breakfast
favily -> family
```

indicate that a few key letters remain incorrectly mapped or that post-processing is not yet connected to a large dictionary.

Exact recovery is more likely when:

- The passage is long.
- The passage resembles the training books.
- Trigram or fourgram context is available.
- Spaces are retained.
- Annealing uses sufficient restarts and iterations.
- Random search is repeated or seeded consistently.
- Post-processing has a sufficiently broad dictionary.

No search result should be described as exact unless it is verified against the known plaintext and key.

## 19. Results and project conclusions

The system demonstrates a complete noisy-channel decryption pipeline:

1. A language model learns how English characters follow one another.
2. A cipher creates a challenging candidate-key problem.
3. Frequency analysis supplies a simple baseline.
4. Hill climbing performs local improvement.
5. Simulated annealing explores worse states to escape local maxima.
6. Post-processing repairs individual words.
7. Experiments compare all components across lengths and representations.
8. A Streamlit app exposes the pipeline to a user.

The project shows that:

- Longer ciphertext makes statistical attacks easier.
- Higher n-gram models use more context.
- Smoothing prevents unseen sequences from destroying the search score.
- Hill climbing can improve a frequency baseline.
- Annealing can escape some local maxima.
- Search quality varies by passage and random initialization.
- Word post-processing is useful but limited without a large dictionary.

## 20. Viva and presentation questions

**Why use a noisy-channel model?**

Encryption is the channel; plaintext is the hidden message. We search for plaintext with high language probability while remaining consistent with the observed ciphertext.

**Why is brute force impossible?**

The key has 26! possible permutations, far beyond exhaustive testing.

**Why sum logs instead of multiplying probabilities?**

Logs avoid underflow and turn long probability products into sums.

**Why smooth probabilities?**

Unseen n-grams otherwise have probability zero and make candidate decryption scores negative infinity.

**Why keep spaces?**

Space is an informative character that marks word boundaries.

**Why remove spaces in an experiment?**

It tests whether the solver can work from continuous letter sequences and measures how much accuracy depends on visible word boundaries.

**Why use hill climbing before annealing?**

Hill climbing is faster and simpler, provides a local-search baseline, and can produce a useful starting key for annealing.

**Why use simulated annealing?**

It accepts selected worse moves and can escape local maxima that trap hill climbing.

**Why use n-grams?**

They provide a compact, explainable, and fast language model that predicts characters from their local context.

**Why use edit distance?**

It repairs short spelling deviations after decryption by measuring the cost of inserting, deleting, or replacing characters.

**What is the hardest part?**

The search is sensitive to random starting keys, passage length, model order, smoothing, and temperature. A highly likely LM result is not necessarily the exact original key.

