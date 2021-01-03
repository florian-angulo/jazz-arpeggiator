import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils



class RVAE(nn.Module):
    N_PITCH = 12
    N_MAIN_QUALITY = 3 # A changer aussi dans data_loader
    N_EXTRA_QUALITY = 3
    N_HIDDEN = 256
    N_LATENT = 16
    N_INPUT = N_PITCH * N_MAIN_QUALITY + N_EXTRA_QUALITY + 3
    rnn_type= 'gru'
    # word_dropout= 0.75
    start_tensor = torch.zeros((N_INPUT,))
    sos_idx= -2
    start_tensor[sos_idx] = 1
    
    end_tensor = torch.zeros((N_INPUT,))
    eos_idx= -1
    end_tensor[eos_idx] = 1
    # pad_idx=
    # unk_idx=

    N_LAYERS=1
    bidirectional=False
    def __init__(self, max_sequence_length):
        
        
        super(RVAE, self).__init__()
        # self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        # self.sos_idx = sos_idx
        # self.eos_idx = eos_idx
        # self.pad_idx = pad_idx
        # self.unk_idx = unk_idx

        # self.latent_size = latent_size

        # self.rnn_type = rnn_type
        # self.bidirectional = bidirectional
        # self.num_layers = num_layers
        # self.hidden_size = hidden_size

        # self.word_dropout_rate = word_dropout
        # self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if self.rnn_type == 'rnn':
            rnn = nn.RNN
        elif self.rnn_type == 'gru':
            rnn = nn.GRU

        else:
            raise ValueError()

        self.encoder_rnn = rnn(self.N_INPUT, self.N_HIDDEN, num_layers=self.N_LAYERS, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(self.N_INPUT, self.N_HIDDEN, num_layers=self.N_LAYERS, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if self.bidirectional else 1) * self.N_LAYERS

        self.hidden2mean = nn.Linear(self.N_HIDDEN * self.hidden_factor, self.N_LATENT)
        self.hidden2logv = nn.Linear(self.N_HIDDEN * self.hidden_factor, self.N_LATENT)
        self.latent2hidden = nn.Linear(self.N_LATENT, self.N_HIDDEN * self.hidden_factor)
        self.outputs2pred = nn.Linear(self.N_HIDDEN * (2 if self.bidirectional else 1), self.N_INPUT)
        

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        
        packed_input = rnn_utils.pack_padded_sequence(input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.N_LAYERS > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.N_HIDDEN*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.N_LATENT])
        if torch.cuda.is_available():
            z = z.cuda()
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.N_LAYERS > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.N_HIDDEN)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
       
        packed_input = rnn_utils.pack_padded_sequence(input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        # logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        # logp = logp.view(b, s, self.embedding.num_embeddings)
        recons = torch.sigmoid(self.outputs2pred(padded_outputs))
        
        return recons, mean, logv, z

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            if torch.cuda.is_available():
                z = torch.randn([batch_size, self.N_LATENT]).cuda()
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.N_LAYERS > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.N_HIDDEN)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(torch.zeros((self.N_INPUT))).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                if torch.cuda.is_available():
                    input_sequence = torch.Tensor(batch_size).fill_(self.start_tensor).long().cuda()

            input_sequence = input_sequence.unsqueeze(1)

            

            output, hidden = self.decoder_rnn(input_sequence, hidden)


            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.end_tensor)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.end_tensor).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
