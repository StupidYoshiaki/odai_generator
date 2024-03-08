import spacy 

# dataset.txt
with open("./data/dataset.txt", "r") as f:
    data = f.read().split("\n")

# 形態素解析
nlp = spacy.load('ja_ginza_electra')

noun_toks = []  

for i, text in enumerate(data):
    print(text)
    text = text.replace("お題:", "")
    doc = nlp(text)
    for tok in doc:  
        if tok.pos_ == 'NOUN' or tok.pos_ == 'PROPN':  
            noun_toks.append(tok)  
            print(tok)

    if i == 20:
        break