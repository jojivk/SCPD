

def convert_word_to_ints(word):
  nword=[]
  nword.append(3)
  for ch in word:
     nword.append(ord(ch))
  nword.append(5)
  return nword

rword = convert_word_to_ints("JMJ")
print(rword)

