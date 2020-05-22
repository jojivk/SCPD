#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        ### JMJ Find size of linear layer
        super(CharDecoder, self).__init__()
        self.hidden_size= hidden_size
        self.char_embedding_size=char_embedding_size

        vocab_len = len(target_vocab.char2id)
        self.vocab_len =vocab_len
        if target_vocab :
          self.target_vocab = target_vocab
          pad_token_idx = target_vocab.char2id['<pad>']
          self.decoderCharEmb = nn.Embedding(vocab_len, char_embedding_size, padding_idx=pad_token_idx)
        #print("JMJ :Constructor (EMB, HID, VOC)", char_embedding_size, self.hidden_size, vocab_len)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, vocab_len, bias=True)
        
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        scores=[]
        X = self.decoderCharEmb(input)

        #(ht,ct) = dec_hidden
        #print("HTCT:", ht.size(), ct.size())
        #print("IN:", input.size(), input.size(-1))
        for Xt in X :
            emb = Xt.unsqueeze(0)
            out, dec_hidden = self.charDecoder(emb, dec_hidden)
            (ht, ct) = dec_hidden
            st = self.char_output_projection(ht)
            #st = F.softmax(st, dim=0)
            scores.append(st)

        scores = torch.stack(scores, dim=0).squeeze(1)
        #print("OUT:", scores.size())
        return scores, dec_hidden
        ### END YOUR CODE 

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        celoss = nn.CrossEntropyLoss()
        inp = char_sequence[:-1]
        Y = char_sequence[1:]
        scores, _ = self.forward(inp, dec_hidden) 
        #scores = F.softmax(scores, dim=-1)
        #print("Scores:", scores.size(), Y.size())
        loss = 0
        n = scores.size(0)
        for i in range(n):
          loss += celoss(scores[i],Y[i])
              
        return loss
        ### END YOUR CODE

    def getCharInput(self, bs, current_chars, device):
        ichars = self.decoderCharEmb(current_chars.long())
        ichars = ichars.unsqueeze(0).contiguous()
        return ichars

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        (ht, ct) = initialStates
        bs = ht.size(1)
        hs = self.char_embedding_size
        output_words =["" for i in range(bs)]
        done = [False for i in range(bs)]
        #print("JMJ :", ct.size(), bs, hs, self.target_vocab.start_of_word)
        hidden_states = initialStates

        curr_char = self.target_vocab.start_of_word
        current_chars = np.empty(bs)
        current_chars.fill(curr_char)
        current_chars = torch.from_numpy(current_chars)

        for t in range(max_length) :
          ichars = self.getCharInput(bs, current_chars, device)
          #print("ICHAR:", ichars.size(), self.char_embedding_size, self.hidden_size)
          out, hidden_states = self.charDecoder(ichars, hidden_states)
          st = self.char_output_projection(out)
          #st = F.softmax(st.squeeze(0), dim=1)
          st = F.softmax(st, dim=2)
          val,ind = torch.max(st.squeeze(0),1)
          current_chars = ind
          alldone=True
          for i in range(bs):
            alldone = alldone and done[i] 
            if current_chars[i] == self.target_vocab.end_of_word or done[i] :
                done[i] =True
                continue
            if current_chars[i] == self.target_vocab.start_of_word :
                continue
            output_words[i] = output_words[i] + self.target_vocab.id2char[current_chars[i].item()]
          
          if alldone:
              break

        return output_words
        ### END YOUR CODE


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

