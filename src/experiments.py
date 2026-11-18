import random
import time 
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cipher import Cipher 
from lm import charNGram
from solvers import (FrequencySolver,HillClimbingSolver,SimulatedAnnealingSolver)

ROOT = Path(__file__).resolve().parent.parent
PASSAGES_DIR = ROOT/"data/test/passages"
CLEAN_DIR = ROOT/"data/clean"
RESULTS_DIR = ROOT/"results"

LENGTHS = (50,100,200,500,1000)
TRIALS=1
RANDOM_SEED = 3

def key_accuracy(actual_key,recovered_key):
    return sum(actual_key[letter] == recovered_key[letter] for letter in "abcdefghijklmnopqrstuvwxyz")/26

def character_accuracy(plaintext,recovered):
    matches = sum(expected == actual for expected,actual in zip(plaintext,recovered))
    return matches/max(len(plaintext),1)

def word_accuracy(plaintext,recovered):
    expected_words = plaintext.split()
    recovered_words = recovered.split()
    matches = sum(expected==actual for expected,actual in zip(expected_words,recovered_words))
    return matches/max(len(expected_words),1)

def train_language_model(n=2,k=0.01):
    train_text = " ".join(
        (CLEAN_DIR/name).read_text(encoding='utf-8')
        for name in (
            "austen-emma.txt",
            "austen-persuasion.txt",
            "austen-sense.txt",
            "chesterton-ball.txt",
            "chesterton-thursday.txt",
        )
    )

    model = charNGram(n=n,k=k)
    model.train(train_text)
    return model

def run_trail(plaintext,solver_name,lm,keep_spaces,cipher):
    if not keep_spaces:
        plaintext = plaintext.replace(" ","")
    actual_key = cipher.generate_key()
    ciphertext = cipher.encrypt(plaintext,actual_key)

    start_time = time.perf_counter()

    if solver_name == "frequency":
        frequency_solver = FrequencySolver()
        recovered_key = frequency_solver.solve(ciphertext)
    elif solver_name == "hill":
        baseline_solver = FrequencySolver()
        baseline_key = baseline_solver.solve(ciphertext)
        hill_solver = HillClimbingSolver()

        recovered_key,_,_,_,_ = hill_solver.solve(ciphertext,lm,restarts=5,iterations=500,start_key=baseline_key)
    elif solver_name == "annealing":
        baseline_solver = FrequencySolver()
        baseline_key = baseline_solver.solve(ciphertext)
        annealing_solver = SimulatedAnnealingSolver()

        recovered_key,_,_,_,_,_ = annealing_solver.solve(
            ciphertext,
            lm,
            restarts=1,
            iterations=500,
            initial_temperature=10.0,
            cooling=0.999,
            start_key=baseline_key,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.perf_counter() - start_time
    recovered = cipher.decrypt(ciphertext,recovered_key)
    return{
        "length"  : len(plaintext),
        "solver": solver_name,
        "n": lm.n,
        "keep_spaces": keep_spaces,
        "key_accuracy": key_accuracy(
            actual_key,
            recovered_key,
        ),
        "character_accuracy": character_accuracy(
            plaintext,
            recovered,
        ),
        "word_accuracy": (
            word_accuracy(plaintext, recovered)
            if keep_spaces
            else 0.0
        ),
        "runtime": elapsed,
    }

def run_experiments():
    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(RANDOM_SEED)

    cipher = Cipher()
    rows = []

    for length in LENGTHS:
        moby_file = PASSAGES_DIR/f"melville-moby_dick_{length}.txt"
        hamlet_file = PASSAGES_DIR/f"shakespeare-hamlet_{length}.txt"

        print(f"Testing length {length}...",flush=True)
        for passage_file in (moby_file,hamlet_file):
            print(f" Passage : {passage_file.name}",flush=True)
            passages = passage_file.read_text(encoding='utf-8').splitlines()

            for trial in range(TRIALS):
                print(f" Trial {trial+1}/{TRIALS}",flush=True)
                plaintext = random.choice(passages)

                for keep_spaces in (True,False):
                    for n in (1,2,3,4):
                        lm = train_language_model(n=n)

                        for solver_name  in ("frequency","hill","annealing"):
                            print(f" solver={solver_name},n{lm.n}, "
                                  f"spaces={keep_spaces}",
                                  flush=True,
                            )
                            row = run_trail(plaintext,solver_name,lm,keep_spaces,cipher)
                            row["trial"] = trial
                            row["passages"]=passage_file.stem
                            rows.append(row)

    dataframe = pd.DataFrame(rows)
    csv_path = RESULTS_DIR/"experiments.csv"
    dataframe.to_csv(csv_path,index=False)

    summary = dataframe.groupby(
        ["length","solver","n","keep_spaces"],
        as_index=False,
    ).mean(numeric_only=True)
    summary_path = RESULTS_DIR/"summary.csv"
    summary.to_csv(summary_path,index=False)

    for keep_spaces in (True,False):
        subset = summary[summary["keep_spaces"] == keep_spaces]

        for metric in ("key_accuracy","character_accuracy","word_accuracy"):
            plt.figure()

            for solver_name in ("frequency","hill","annealing"):
                solver_subset = subset[subset["solver"] == solver_name]
                plt.plot(
                    solver_subset["length"],
                    solver_subset[metric],
                    marker="o",
                    label=solver_name,
                )

            plt.xlabel("Ciphertext length")
            plt.ylabel(metric.replace("_"," ").title())
            plt.title(
                f"Accuracy vs length "
                f"(spaces={'kept' if keep_spaces else 'removed'})"
            )
            plt.legend()
            plt.grid(True)

            filename = (
                f"{metric}_"
                f"{'spaces' if keep_spaces else 'no_spaces'}"
                ".png"
            )
            plt.savefig(RESULTS_DIR/filename,bbox_inches="tight")
            plt.close()

    print(f"Results saved to {csv_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_experiments()

                            
