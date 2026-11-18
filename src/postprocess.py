from collections import Counter 
from difflib import SequenceMatcher
import re 
import string

COMMON_WORDS = """
the of and to a in that is was we for it with as be at by this had not but from or have an they which one you were all went can their what said there been would will if some so no king queen castle england man men time day night people
""".split()

WORD_FREQUENCY = Counter(COMMON_WORDS)

def levenshtein_distance(first,second):
    if len(first) < len(second):
        return levenshtein_distance(second,first)
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first,start=1):
        current = [i]

        for j,second_char in enumerate(second,start=1):
            insert = current[j-1]+1
            delete = previous[j]+1
            substitute = previous[j-1]+(first_char != second_char)
            current.append(min(insert,delete,substitute))

        previous = current

    return previous[-1]

def is_known_word(word):
    return word in WORD_FREQUENCY

def correct_word(word,candidates=None,max_distance=2):
    word = word.lower()

    if is_known_word(word):
        return word

    if candidates is None:
        candidates = WORD_FREQUENCY.keys()

    ranked=[]

    for candidate in candidates:
        distance = levenshtein_distance(word,candidate)

        if distance <= max_distance:
            ranked.append((distance,-WORD_FREQUENCY[candidate],-SequenceMatcher(None,word,candidate).ratio(),candidate))

    if not ranked:
        return word

    ranked.sort()
    return ranked[0][3]

def postprocess(text,candidates=None):
    def replace(match):
        word = match.group(0)
        return correct_word(word,candidates=candidates)
    return re.sub(r"[a-z]+",replace,text.lower())

if __name__=="__main__":
    sample = "the king of england went to the castle"

    corrected = postprocess(sample)
    print("Original : ",sample)
    print("Corrected : ",corrected)

    assert postprocess("the king") == "the king"
    assert postprocess("englad") == "england"
    assert postprocess("castel") == "castle"

    print("post-processing checkpoint : PASSED")