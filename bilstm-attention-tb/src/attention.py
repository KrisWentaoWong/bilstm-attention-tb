# -*- coding: utf-8 -*
import logging
import dynet as dy


class AttentionNetwork:

    def __init__(self, model, input_dim, output_dim, rnn_dropout_rate=0.1):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.w1 = self.model.add_parameters((self.output_dim, self.output_dim))
        self.v = self.model.add_parameters((1, self.output_dim))

    def __call__(self, input_sequence):
        """
        Adapted from https://talbaumel.github.io/blog/attention/
        :param input_sequence:
        :return:
        """

        # 1.) Feed the input sequence through the encoder and get
        #     the hidden states for every input word.
        encoded_sequence = input_sequence #self.encoder.get_encoded_sequence(input_sequence)

        # 2.) Calculate the attention weight for every vector in the encoding
        attention_weights = []
        for encoded_token in encoded_sequence:
            attention_weight = self.v * dy.tanh(self.w1 * encoded_token)
            attention_weights.append(attention_weight)

        # Compute the softmax over the computed weights
        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        # 3.) Compute a weighted sum over the input vectors
        tokens_and_weights = zip(encoded_sequence, attention_weights)
        sequence_encoding = dy.esum([
            encoded_token * attention_weight
            for encoded_token, attention_weight in tokens_and_weights
        ])

        return sequence_encoding
