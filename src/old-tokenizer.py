# Creation d'un tokeniser mot/mot 
import re

#Encodeur du dataset
#Etape 1: Tokeniser le texte
# On affiche quelques petites choses de base pour voir si tout va bien
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
#print("Nombre total de caracteres:", len(raw_text))
#print("Affichage des 99 premiers caractères:",raw_text[:99])

# Nous devons maintenant créer une lliste de caracteres pour chaque phrase du texte. 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

#print("Text splité:",preprocessed[:30])

#Etape 2: transformer les tokens en tokens IDs
# On crée un vocabulaire de tokens uniques en faisant un dictionnaire
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#On prend tous les tokens et on ajoute un token spécial pour les mots qui ne sont pas dans notre vocabulaire    
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}


class SimpleTokenizer:
    # Initialisation avec le vocabulaire        
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    #Encodeur du texte/prompt 
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Gestion des tokens inconnus
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    #Decodeur des token IDs en texte/prompt
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizer(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(tokenizer.decode(tokenizer.encode(text)))

