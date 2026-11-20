import streamlit as st
from pathlib import Path
import sys
import re
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"src"))

from cipher import Cipher
from lm import charNGram
from solvers import (
    FrequencySolver,
    HillClimbingSolver,
    SimulatedAnnealingSolver
)

"""
call me ishmael some years ago never mind how long precisely having 
little or no money in my purse and nothing particular to interest me on 
shore i thought i would sail about a little and see the watery part of 
the world it is a way i have of driving off the spleen and regulating the 
circulation whenever i find myself growing grim about the mouth whenever it 
is a damp drizzly november in my soul whenever i find myself involuntarily 
pausing before coffin warehouses and bringing up the rear of every funeral 
i meet and especially whenever my hypos get such an upper hand of me that 
it requires a strong moral principle to prevent me from deliberately stepping 
into the street and methodically knocking peoples hats off then i account it 
high time to get to sea as soon as i can
"""

@st.cache_resource
def load_language_model(n=2):
    clean_dir = ROOT/"data/clean"
    train_text = " ".join(
        (clean_dir/name).read_text(encoding='utf-8') for name in (
            "austen-emma.txt",
            "austen-persuasion.txt",
            "austen-sense.txt",
            "chesterton-ball.txt",
            "chesterton-thursday.txt",
        )
    )

    model = charNGram(n=n,k=0.01)
    model.train(train_text)
    return model

st.set_page_config(page_title="Substitution Cipher Solver",layout="wide")
st.title("Substitution Cipher Solver")
text = st.text_area("Plaintext","the king of england went to the castle")

n = st.selectbox("Language model n",(1,2,3,4),index=1)

solver_name = st.selectbox(
    "Solver",
    ("Frequency baseline","Hill Climbing","Simulated annealing"),
)

cipher = Cipher()
if "key" not in st.session_state:
    st.session_state.key = cipher.generate_key()

if  st.button("Generate a new key"):
    st.session_state.key = cipher.generate_key()

plaintext = text.lower()
plaintext = re.sub(r"[^a-z]+"," ",plaintext)
plaintext = re.sub(r"\s+"," ",plaintext).strip()
ciphertext = cipher.encrypt(plaintext,st.session_state.key)


#SoLVE
lm = load_language_model(n=n)

if st.button("Decrypt"):
    if not plaintext:
        st.warning("Enter some text containing letters.")
        st.stop()

    frequency = FrequencySolver()
    baseline_key = frequency.solve(ciphertext)
    score = None
    elapsed = None

    if solver_name == "Frequency baseline":
        solved_key = baseline_key
    elif solver_name == "Hill Climbing":
        with st.spinner("Running hill climbing..."):
            solver = HillClimbingSolver()
            random.seed(42)
            solved_key,score,restarts,improvements,elapsed = solver.solve(
                ciphertext,
                lm,
                restarts=10,
                iterations=500,
                start_key=baseline_key,
            )
    else:
        with st.spinner("Running simulated annealing..."):
            solver = SimulatedAnnealingSolver()
            random.seed(42)
            solved_key,score,restarts,improvements,worse_moves,elapsed = solver.solve(
                ciphertext,
                lm,
                restarts=5,
                iterations=5000,
                initial_temperature=10.0,
                cooling=0.999,
                start_key=baseline_key,
            )

    recovered = cipher.decrypt(ciphertext,solved_key)
    st.subheader("Ciphertext")
    st.code(ciphertext)

    st.subheader("Recovered plaintext")
    st.code(recovered)

    if solver_name == "Frequency baseline":
        st.caption("Frequency baseline completed.")
    else:
        st.caption(
            f"Solver: {solver_name} | "
            f"Log probability: {score:.2f} | "
            f"Runtime: {elapsed:.2f}s"
        )
