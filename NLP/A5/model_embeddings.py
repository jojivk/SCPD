#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 
class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        pad_token_idx = vocab['<pad>']
        #if not use_char_encodings:
        #  self.embeddings = nn.Embedding(len(vocab.word2id), embed_size, padding_idx=pad_token_idx)
        #else:
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(embed_size, 5) # What is k JMJ
        self.highway =Highway(embed_size)
        self.dropout = nn.Dropout(0.3)

        #self.source = SourceEmbedding(echar, embed_size, ksize)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        xpadded = input_tensor
        xemb = self.embeddings(xpadded)
        #if not self.use_char_encodings :
        #  return xemb 
        xreshaped = xemb.permute(0,1,3,2)
        xconv_out = self.cnn(xreshaped)
        xhighway = self.highway(xconv_out)
        xword_emb= self.dropout(xhighway)
        return xword_emb  #.permute(1,0,2)

        ### END YOUR CODE
