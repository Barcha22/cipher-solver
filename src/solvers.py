from collections import Counter
from string import ascii_lowercase
from cipher import Cipher
from lm import charNGram
from pathlib import Path
import random
import time
import math

ENGLISH_FREQUENCY = "etaoinshrdlcumwfgypbvkjxqz" #english letters from most to least frequent 

#we are not using spaces in this ranking ->
class FrequencySolver:
    def __init__(self):
        pass

    def solve(self,ciphertext):
        frequencies = Counter(
            character for character in ciphertext
            if character in ascii_lowercase
        )

        cipher_ranking = [character for character,_ in frequencies.most_common()]

        cipher_ranking.extend(
            character for character in ascii_lowercase
            if character not in cipher_ranking
        )

        cipher_to_plain = dict(zip(cipher_ranking,ENGLISH_FREQUENCY))

        plaintext_to_cipher = {plaintext:cipher for cipher,plaintext in cipher_to_plain.items()}

        return plaintext_to_cipher


class HillClimbingSolver:
    def __init__(self):
        self.cipher = Cipher()

    def climb(self, ciphertext, lm, start_key, iterations=1000):
        current_key = start_key.copy()
        current_text = self.cipher.decrypt(ciphertext, current_key)
        current_score = lm.logprob(current_text)
        improvements = 0

        for iteration in range(1, iterations + 1):
            first, second = random.sample(list(ascii_lowercase), 2)
            candidate_key = current_key.copy()
            candidate_key[first], candidate_key[second] = (
                candidate_key[second],
                candidate_key[first],
            )

            candidate_text = self.cipher.decrypt(ciphertext, candidate_key)
            candidate_score = lm.logprob(candidate_text)

            if candidate_score > current_score:
                current_key = candidate_key
                current_score = candidate_score
                improvements += 1

        return current_key, current_score, improvements

    def solve(self, ciphertext, lm, restarts=10, iterations=1000, start_key=None):
        start_time = time.perf_counter()
        best_key = None
        best_score = float("-inf")
        total_improvements = 0

        for restart in range(restarts):
            if restart == 0 and start_key is not None:
                restart_key = start_key.copy()
            else:
                restart_key = self.cipher.generate_key()

            candidate_key, candidate_score, improvements = self.climb(
                ciphertext,
                lm,
                restart_key,
                iterations,
            )
            total_improvements += improvements

            if candidate_score > best_score:
                best_key = candidate_key
                best_score = candidate_score

        elapsed = time.perf_counter() - start_time
        return best_key, best_score, restarts, total_improvements, elapsed


class SimulatedAnnealingSolver:
    def __init__(self):
        self.cipher = Cipher()

    def solve(self,ciphertext,lm,restarts=5,iterations=5000,initial_temperature=1.0,cooling=0.995,start_key=None):
        start_time = time.perf_counter()
        best_key = None
        best_score = float("-inf")
        total_improvements = 0
        total_worse_moves = 0

        for restart in range(restarts):
            if restart == 0 and start_key is not None:
                current_key = start_key.copy()
            else:
                current_key = self.cipher.generate_key()

            current_text = self.cipher.decrypt(ciphertext,current_key)
            current_score = lm.logprob(current_text)

            restart_best_key = current_key.copy()
            restart_best_score = current_score
            temperature = initial_temperature

            for iteration in range(iterations):
                first,second = random.sample(list(ascii_lowercase),2)

                candidate_key = current_key.copy()
                candidate_key[first], candidate_key[second] = (
                    candidate_key[second],
                    candidate_key[first]
                )

                candidate_text = self.cipher.decrypt(ciphertext,candidate_key)
                candidate_score = lm.logprob(candidate_text)

                delta = candidate_score - current_score
                accept = delta >= 0

                if delta < 0:
                    probability = math.exp(delta/temperature)
                    accept = random.random() < probability
                    if accept:
                        total_worse_moves +=1

                if accept:
                    current_key = candidate_key
                    current_score = candidate_score

                    if candidate_score > restart_best_score:
                        restart_best_key = candidate_key.copy()
                        restart_best_score = candidate_score
                        total_improvements +=1

                temperature = max(temperature*cooling, 1e-8)
            if restart_best_score > best_score:
                best_key = restart_best_key.copy()
                best_score = restart_best_score
        elapsed = time.perf_counter() - start_time
        return (best_key,best_score,restarts,total_improvements,total_worse_moves,elapsed)


if __name__ == "__main__":
    cipher = Cipher()
    solver = FrequencySolver()

    root = Path(__file__).resolve().parent.parent
    passages_dir = root / "data/test/passages"

    for length in (50, 100, 200, 500, 1000):
        key_accuracy_total = 0.0
        character_accuracy_total = 0.0
        passage_count = 0

        for passage_file in passages_dir.glob(f"*_{length}.txt"):
            for plaintext in passage_file.read_text(
                encoding='utf-8'
            ).splitlines():

                actual_key = cipher.generate_key()
                ciphertext = cipher.encrypt(plaintext, actual_key)

                baseline_key = solver.solve(ciphertext)
                recovered = cipher.decrypt(ciphertext, baseline_key)

                key_matches = sum(
                    1
                    for plaintext_letter in ascii_lowercase
                    if actual_key[plaintext_letter]
                    == baseline_key[plaintext_letter]
                )

                character_matches = sum(
                    expected == actual
                    for expected, actual in zip(plaintext, recovered)
                )

                key_accuracy_total += key_matches / 26
                character_accuracy_total += character_matches / len(plaintext)
                passage_count += 1

        print(
            f"Length {length}: "
            f"key accuracy={key_accuracy_total / passage_count:.3f}, "
            f"character accuracy={character_accuracy_total / passage_count:.3f}"
        )

    # One visible example for the first Moby Dick passage.
    example_file = passages_dir / "melville-moby_dick_100.txt"
    plaintext = example_file.read_text(encoding="utf-8").splitlines()[0]

    actual_key = cipher.generate_key()
    ciphertext = cipher.encrypt(plaintext, actual_key)
    baseline_key = solver.solve(ciphertext)
    recovered = cipher.decrypt(ciphertext, baseline_key)

    print("\nExample:")
    print("Plaintext:", plaintext)
    print("Ciphertext:", ciphertext)
    print("Recovered:", recovered)

    # Hill-climbing checkpoint.
    root = Path(__file__).resolve().parent.parent
    clean_dir = root / "data/clean"
    train_text = " ".join(
        (clean_dir / name).read_text(encoding="utf-8")
        for name in (
            "austen-emma.txt",
            "austen-persuasion.txt",
            "austen-sense.txt",
            "chesterton-ball.txt",
            "chesterton-thursday.txt",
        )
    )
    lm = charNGram(n=2, k=0.01)
    lm.train(train_text)

    hc_file = passages_dir / "melville-moby_dick_200.txt"
    plaintext = hc_file.read_text(encoding="utf-8").splitlines()[0]
    actual_key = cipher.generate_key()
    ciphertext = cipher.encrypt(plaintext, actual_key)
    baseline_key = solver.solve(ciphertext)

    random.seed(42)
    hill_solver = HillClimbingSolver()
    best_key, best_score, restarts, improvements, elapsed = hill_solver.solve(
        ciphertext,
        lm,
        restarts=10,
        iterations=500,
        start_key=baseline_key,
    )
    recovered = cipher.decrypt(ciphertext, best_key)

    key_accuracy = sum(
        actual_key[letter] == best_key[letter]
        for letter in ascii_lowercase
    ) / 26
    character_accuracy = sum(
        expected == actual
        for expected, actual in zip(plaintext, recovered)
    ) / len(plaintext)

    print("\nHill climbing:")
    print("Recovered:", recovered)
    print(f"Score: {best_score:.2f}")
    print(f"Restarts: {restarts}")
    print(f"Accepted swaps: {improvements}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Key accuracy: {key_accuracy:.3f}")
    print(f"Character accuracy: {character_accuracy:.3f}")

    #Simulated Annealing
    random.seed(42)
    sa_solver = SimulatedAnnealingSolver()

    (
        best_key,
        best_score,
        restarts,
        improvements,
        worse_moves,
        elapsed,
    ) = sa_solver.solve(
        ciphertext,
        lm,
        restarts=5,
        iterations=5000,
        initial_temperature=10.0,
        cooling=0.999,
        start_key=baseline_key,
    )

    recovered = cipher.decrypt(ciphertext, best_key)

    key_accuracy = sum(
        actual_key[letter] == best_key[letter]
        for letter in ascii_lowercase
    ) / 26

    character_accuracy = sum(
        expected == actual
        for expected, actual in zip(plaintext, recovered)
    ) / len(plaintext)

    print("\nSimulated annealing:")
    print("Recovered:", recovered)
    print(f"Score: {best_score:.2f}")
    print(f"Restarts: {restarts}")
    print(f"Best-key improvements: {improvements}")
    print(f"Accepted worse swaps: {worse_moves}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Key accuracy: {key_accuracy:.3f}")
    print(f"Character accuracy: {character_accuracy:.3f}")

    print("\nSimulated annealing tuning:")
    print("temperature  cooling  key_acc  char_acc  accepted_worse  time")

    for initial_temperature in (0.1, 1.0, 10.0):
        for cooling in (0.99, 0.995, 0.999):
            random.seed(42)
            tuning_solver = SimulatedAnnealingSolver()
            tuning_result = tuning_solver.solve(
                ciphertext,
                lm,
                restarts=3,
                iterations=2000,
                initial_temperature=initial_temperature,
                cooling=cooling,
                start_key=baseline_key,
            )
            tuning_key = tuning_result[0]
            tuning_worse_moves = tuning_result[4]
            tuning_elapsed = tuning_result[5]
            tuning_recovered = cipher.decrypt(ciphertext, tuning_key)

            tuning_key_accuracy = sum(
                actual_key[letter] == tuning_key[letter]
                for letter in ascii_lowercase
            ) / 26
            tuning_character_accuracy = sum(
                expected == actual
                for expected, actual in zip(plaintext, tuning_recovered)
            ) / len(plaintext)

            print(
                f"{initial_temperature:<12.1f} "
                f"{cooling:<8.3f} "
                f"{tuning_key_accuracy:<8.3f} "
                f"{tuning_character_accuracy:<9.3f} "
                f"{tuning_worse_moves:<15d} "
                f"{tuning_elapsed:.2f}s"
            )