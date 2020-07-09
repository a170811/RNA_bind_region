from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed
import numpy as np

from utils.dataset import build_data


class AutoEncoder:

    def train(self, input_x, hidden_dim=50):

        target_x = [f'\t{seq}\n' for seq in x]

        input_charactors = sorted(list(set(''.join(input_x))))
        target_charactors = sorted(list(set(''.join(target_x))))
        num_encoder_token = len(input_charactors)
        num_decoder_token = len(target_charactors)
        max_encoder_seq_length = max(map(len, input_x))
        max_decoder_seq_length = max(map(len, target_x))

        print(f'input_charactor: {input_charactors}')
        print(f'target_charactors: {target_charactors}')
        print(f'num_encoder_token: {num_encoder_token}')
        print(f'num_decoder_token: {num_decoder_token}')
        print(f'max_encoder_seq_length: {max_encoder_seq_length}')
        print(f'max_decoder_seq_length: {max_decoder_seq_length}')

        input_token2idx = {token: i for i, token in enumerate(input_charactors)}
        target_token2idx = {token: i for i, token in enumerate(target_charactors)}

        padding = lambda seq, max_len: seq + '0'*(max_len-len(seq))
        encoder_input_data = np.array([list(padding(seq, max_encoder_seq_length)) for seq in input_x])
        decoder_input_data = np.array([list(padding(seq[1:], max_decoder_seq_length)) for seq in target_x]) # decoder input has not '\t' token
        decoder_target_data = np.array([list(padding(seq, max_decoder_seq_length)) for seq in target_x])

        print(encoder_input_data.shape)
        print(decoder_input_data.shape)
        print(decoder_target_data.shape)

        encoder_embedding_layer = Embedding(input_dim=num_encoder_token, output_dim=hidden,\
                                            input_length=max_encoder_seq_length)
        decoder_embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN)

        encoder_inputs = Input(shape=(None, num_encoder_token))
        encoder_LSTM = LSTM(hidden_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

        decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
        decoder_embedding = embedding_layer(decoder_inputs)
        decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

        # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
        outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], outputs)


        self.model = seq2seq_model_builder(hidden_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

if '__main__' == __name__:


    # model = seq2seq_model_builder(HIDDEN_DIM=50)
    # model.summary()

    data_dir = './data/raw'
    x, _ = build_data(data_dir)
    x = x.flatten()
    ae = AutoEncoder()
    ae.train(x)

