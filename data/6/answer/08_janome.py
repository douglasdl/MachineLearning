from janome.tokenizer import Tokenizer

t = Tokenizer()

s = 'これらは、なんと良い本なのだろう。'
t.tokenize(s)

str = []
for token in t.tokenize(s,wakati=True):
    str.append(token)
print(" ".join(str))