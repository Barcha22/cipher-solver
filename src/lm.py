#  Why smoothing? The −inf catastrophe. With a trigram model, a wrong decryption will certainly produce some trigram you never saw in training, like xqj. Plain counting gives it probability 0 → log(0) = −inf → the whole score is −inf. Now every candidate key scores −inf and the solver is blind. Add-k smoothing fixes it: pretend every event was seen k times (k ≈ 0.01–1):
#  Why log probabilities? Multiplying 200 probabilities like 0.03 gives ~10⁻³⁰⁵ — float64 underflows to exactly 0 around 10⁻³⁰⁸. Scores would collapse to 0 and be uncomparable. Taking log turns the product into a sum, which stays finite: score = Σ log P(cᵢ | contextᵢ). More negative = worse.

import math
from collections import Counter
from data import Data
from pathlib  import Path

class charNGram:
    def __init__(self,n=2,k=0.01):
        self.n,self.k,self.V=n,k,27
        self.ngram_count = Counter()
        self.context_count = Counter()

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


if __name__ == "__main__"():
    data = Data()
    assert data.check_files(), "data check failed in data.py"

    root = Path(__file__).resolve().parent.parent
    

