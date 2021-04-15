#/2021/04/15

##Dealing with the text data.

##Read data
import torch
with open('../data/jane-austen/1342-0.txt', encoding='utf8') as f:
    text = f.read()

##split the text data line by line
lines = text.split('\n')

##randomly picked a line
line = lines[200]

letter_t = torch.zeros(len(line), 128)
print(letter_t.shape)

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

#print(letter_t)

###For whole lines
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list
words_in_line = clean_words(line)
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
