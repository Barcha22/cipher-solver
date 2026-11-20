import streamlit as st
from pathlib import Path
import sys
import random 
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"src"))

from cipher import Cipher
from lm import charNGram
from solvers import (
    FrequencySolver,
    HillClimbingSolver,
    SimulatedAnnealingSolver
)


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

solver_name = st.selectbox("Solver",("Frequency basline","Hill Climbing","Simulated annealing"))

cipher = Cipher()
if "key" not in st.session_state:
    st.session_state.key = cipher.generate_key()

if  st.button("Generate a new key"):
    st.session_state.key = cipher.generate_key()

plaintext = text.lower()
ciphertext = cipher.encrypt(plaintext,st.session_state.key)


#SoLVE
lm = load_language_model(n=n)

if st.button("Decrypt"):
    frequency = FrequencySolver()
    baseline_key = frequency.solve(ciphertext)

    if solver_name == "Frequency baseline":
        solved_key = baseline_key
    elif solver_name == "Hill Climbing":
        solver = HillClimbingSolver()
        solved_key,score,_,_,_ = solver.solve(ciphertext,lm,restarts=10,iterations=500,start_key=baseline_key)
    else:
        solver = SimulatedAnnealingSolver()
        solved_key,score,_,_,_,_ = solver.solve(ciphertext,lm,restarts=5,iterations=5000,initial_temperature=10.0,cooling=0.999)

    recovered = cipher.decrypt(ciphertext,solved_key)
    st.subheader("Ciphertext")
    st.code(ciphertext)

    st.subheader("Recovered plaintext")
    st.code(recovered)
    