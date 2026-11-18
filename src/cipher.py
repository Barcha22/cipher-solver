import random
import string

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

class Cipher():
    def __init__(self):
        pass

    def generate_key(self):
        letters = list(string.ascii_lowercase)
        shuffled = letters.copy()
        random.shuffle(shuffled)
        return dict(zip(letters,shuffled))

    def encrypt(self,text,key):
        return "".join(key.get(char.lower(),char) for char in text.lower())

    def decrypt(self,text,key):
        inverse = {cipher:plain for plain, cipher in key.items()}
        return "".join(inverse.get(char.lower(),char) for char in text.lower())

    
if __name__=="__main__":
    cipher = Cipher()
    key = cipher.generate_key()

    plaintext = "the king of england went to the castle"
    ciphertext = cipher.encrypt(plaintext,key)
    recovered = cipher.decrypt(ciphertext,key)

    print("Key : ",key)
    print("Plain text : ",plaintext)
    print("Cipher text : ",ciphertext)
    print("Recovered : ",recovered)
    assert len(key) == 26
    assert set(key.keys())==set(string.ascii_lowercase)
    assert set(key.values())==set(string.ascii_lowercase)    
    assert recovered == plaintext
    print("Cipher round: PASSED")