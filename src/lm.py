#  Why smoothing? The −inf catastrophe. With a trigram model, a wrong decryption will certainly produce some trigram you never saw in training, like xqj. Plain counting gives it probability 0 → log(0) = −inf → the whole score is −inf. Now every candidate key scores −inf and the solver is blind. Add-k smoothing fixes it: pretend every event was seen k times (k ≈ 0.01–1):
#  Why log probabilities? Multiplying 200 probabilities like 0.03 gives ~10⁻³⁰⁵ — float64 underflows to exactly 0 around 10⁻³⁰⁸. Scores would collapse to 0 and be uncomparable. Taking log turns the product into a sum, which stays finite: score = Σ log P(cᵢ | contextᵢ). More negative = worse.

import math
from collections import Counter
from data import Data
from pathlib  import Path

class charNGram:
    def __init__(self,n=2,k=0.01):
        self.n,self.k,self.V=n,k,27
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def train(self,text):
        for i in range(len(text) - self.n + 1):
            self.ngram_counts[text[i:i+self.n]] += 1
            self.context_counts[text[i:i+self.n-1]] += 1

    def logprob(self,text):
        total = 0.0
        for i in range(len(text) - self.n + 1):
            g = text[i:i+self.n]
            total += math.log((self.ngram_counts[g] + self.k) /
                              (self.context_counts[g[:-1]] + self.k * self.V))
        return total

    def perplexity(self,text):
        n_events = len(text) - self.n + 1
        return math.exp(-self.logprob(text) / n_events)

    @staticmethod
    def interpolated_logprob(text, models, weights):
        if len(weights) != len(models):
            raise ValueError("Each model needs an interpolation weight")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Interpolation weights must sum to 1")
        total = 0.0
        for i in range(len(text)):
            probabilities = []
            available_weights = []

            for model, weight in zip(models, weights):
                n = model.n
                if i + 1 < n:
                    continue
                gram = text[i - n + 1:i + 1]
                context = gram[:-1]
                probability = (
                    (model.ngram_counts[gram] + model.k) /
                    (model.context_counts[context] + model.k * model.V)
                )
                probabilities.append(probability)
                available_weights.append(weight)

            weight_sum = sum(available_weights)
            combined = sum(
                weight * probability
                for weight, probability in zip(available_weights, probabilities)
            ) / weight_sum
            total += math.log(combined)
        return total

    @staticmethod
    def interpolated_perplexity(text, models, weights):
        logprob = charNGram.interpolated_logprob(text, models, weights)
        if not text:
            raise ValueError("Cannot calculate perplexity on empty text")
        return math.exp(-logprob / len(text))


if __name__ == "__main__":
    data = Data()
    assert data.check_files(), "data check failed in data.py"

    root = Path(__file__).resolve().parent.parent
    texts = [(root/"data/clean"/name).read_text(encoding='utf-8') for name in sorted(data.TRAIN_BOOKS)]

    train_text = " ".join(texts)
    print(f"Training on {len(train_text):,} chars from {len(texts)} books")
    
    uni = charNGram(n=1)
    uni.train(train_text)

    bi = charNGram(n=2)
    bi.train(train_text)

    tri = charNGram(n=3)
    tri.train(train_text)

    four = charNGram(n=4)
    four.train(train_text)

    print(f"Top 10 bigrams : {bi.ngram_counts.most_common(10)}")

    #condition probabilities .e.g, for th
    alphabet = "abcdefghijklmnopqrstuvwxyz "

    p_sum = sum((bi.ngram_counts["t"+c] + bi.k) /
                (bi.context_counts["t"] + bi.k * bi.V) for c in alphabet)
    print("Sum of P(c|'t') over all 27 chars:", p_sum)

    # English vs gibberish -> perplexities
    eng = "the king of england went to the castle"
    gib = "xqz jwvkp ghfm blnqzj wvkp ghfmbln"
    print(f"english   logprob={bi.logprob(eng):.1f}  ppl={bi.perplexity(eng):.1f}")
    print(f"gibberish logprob={bi.logprob(gib):.1f}  ppl={bi.perplexity(gib):.1f}")

    # 4 - first perplexities: train slice vs unseen moby
    moby = (root / "data/test/melville-moby_dick.txt").read_text(encoding="utf-8")
    for model, name in ((uni, "unigram"), (bi, "bigram")):
        print(f"{name}: ppl(train 10k)={model.perplexity(train_text[:10000]):.2f}  "
              f"ppl(moby 1000)={model.perplexity(moby[:1000]):.2f}")

    # held out perplexities
    models = [uni, bi, tri, four]
    weights = [0.05, 0.15, 0.30, 0.50]
    
    moby = (
    root / "data/test/melville-moby_dick.txt").read_text(encoding="utf-8")

    for n in range(1, 5):
        model = charNGram(n=n, k=0.01)
        model.train(train_text)
        print(
            f"n={n}, k=0.01, "
            f"Moby perplexity={model.perplexity(moby[:1000]):.2f}"
        )

    moby_logprob = charNGram.interpolated_logprob(
        moby[:1000],
        models,
        weights,
    )
    moby_events = len(moby[:1000])
    moby_ppl = math.exp(-moby_logprob / moby_events)

    print(f"Interpolated PPL: {moby_ppl:.2f}")

    hamlet = (root / "data/test/shakespeare-hamlet.txt").read_text(encoding="utf-8")
    held_out_texts = (("moby", moby[:1000]), ("hamlet", hamlet[:1000]))
    smoothing_values = (0.01, 0.1, 0.5, 1.0)

    print("\nPerplexity table (n x smoothing)")
    print(f"{'held-out':<10} {'k':<6} {'uni':>8} {'bi':>8} {'tri':>8} {'four':>8}")
    for book_name, held_out in held_out_texts:
        for k in smoothing_values:
            grid_models = [charNGram(n=n, k=k) for n in range(1, 5)]
            for grid_model in grid_models:
                grid_model.train(train_text)
            print(
                f"{book_name:<10} {k:<6.2f} "
                f"{grid_models[0].perplexity(held_out):8.2f} "
                f"{grid_models[1].perplexity(held_out):8.2f} "
                f"{grid_models[2].perplexity(held_out):8.2f} "
                f"{grid_models[3].perplexity(held_out):8.2f}"
            )
