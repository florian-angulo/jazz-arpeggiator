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
    word_dropout_rate = 0.75  
    
    N_LAYERS=1
    bidirectional=False
    
    no_chords = torch.zeros(N_INPUT).cuda()
    no_chords[-1] = 1
    def __init__(self, max_sequence_length,rnn_type="gru",bidirectional=False):
        
        
        super(RVAE, self).__init__()
        
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.max_sequence_length = max_sequence_length

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
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
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            d1, d2, _ = input_sequence.size()
            prob = torch.rand((d1,d2))
            prob[:,0] = 1 # Not to change start token
            if torch.cuda.is_available():
                prob=prob.cuda()
           
            decoder_input_sequence = input_sequence.clone()
            
            decoder_input_sequence[prob < self.word_dropout_rate] = self.no_chords

        
        packed_input = rnn_utils.pack_padded_sequence(decoder_input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to one_hot representation
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
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).float()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).float()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).float()
        
        generations = self.tensor(batch_size, self.max_sequence_length,self.N_INPUT).fill_(0).float()
        
        input_sequence = self.tensor(batch_size, self.max_sequence_length, self.N_INPUT).fill_(0).float().cuda()
        input_sequence[:, 0, -3] = 1 # Initialize START on every beginnings of the sentences
        input_sequence[:, 1:, -1] = 1 # Initialize N on the rest of the sentences
        
        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:



            output, hidden = self.decoder_rnn(input_sequence[:, :t+1], hidden)

            # create new input_sequence from output

            proba_output = torch.sigmoid(self.outputs2pred(output))
            for i in range(n):
                input_sequence[i,t] = sentence_to_tensor(tensor_to_sentence(proba_output[i], t+1))[t].float()
            
            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)
            
            # update gloabl running sequence
            sequence_mask[sequence_running.long()] = (input_sequence[:, t, -2] != 1)
            sequence_running = sequence_idx.masked_select(sequence_mask)
            # update local running sequences
            running_mask = (input_sequence[:, t, -2] != 1)
            running_seqs = running_seqs.masked_select(running_mask)
            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs.long()]
                hidden = hidden[:, running_seqs.long()]
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).float()
                

            t += 1

        return generations, z

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs.long()]
        # update token at position t
        running_latest[:,:t+1] = sample[:,:t+1].data
        # save back
        save_to[running_seqs.long()] = running_latest

        return save_to
