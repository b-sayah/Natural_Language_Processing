# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa

from allennlp.training.metrics import BLEU  # noqa


class Seq2Seq():
    def __init__(self, vocab, cfg, device, name=None, bi=True, att=True, batch_norm=True,
        teach_forc_ratio=0.5, patience=3, dropout=0.0, write_idx=3):
        self.device = device
        self.model = Seq2SeqArch(vocab, cfg, device,
                                 bi=bi, att=att, batch_norm=batch_norm, teach_forc_ratio=teach_forc_ratio, dropout=dropout)
        self.vocab = vocab
        self.name = name if name else 'seq2seq'
        self.cfg = cfg
        self.write_idx = write_idx

        # Evaluation metrics
        self.bleu = BLEU(exclude_indices=set([0]))  # Exclude padding

        # Logging variables
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        self.train_bleu, self.valid_bleu, self.test_bleu = [], [], []
        self.patience = patience  # For early stopping
        self.outputs = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_outputs(self, log_dir):
        '''
        Log the test outputs to a txt.
        '''
        with open(os.path.join(log_dir, '{0}_outputs_{1}.txt'.format(self.name, self.write_idx)), 'w') as fp:
            for line in self.outputs:
                fp.write('{0}\n'.format(line))

    def log_learning_curves(self, log_dir, graph=True):
        '''
        Logs the learning curve info to a csv.
        '''
        header = 'epoch,train_loss,valid_loss'
        num_epochs = len(self.train_losses)
        with open(os.path.join(log_dir, '{0}_learning_curves.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_losses[e], self.valid_losses[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_losses, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_losses, color='red', label='Valid')
            plt.title('Cross-entropy loss over training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_learning_curves.png'.format(self.name)))
            plt.clf()

    def log_metrics(self, log_dir, graph=True):
        '''
        Logs evaluation metrics (BLEU, etc.) to a csv.
        '''
        header = 'epoch,train_bleu,valid_bleu'
        num_epochs = len(self.train_bleu)
        with open(os.path.join(log_dir, '{0}_metrics.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_bleu[e], self.valid_bleu[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_bleu, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_bleu, color='red', label='Valid')
            plt.title('BLEU score over training')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_metrics.png'.format(self.name)))
            plt.clf()

    def generate(self, qc_sentence):
        print('inp qc:', qc_sentence)
        qc_words = ['BOS'] + qc_sentence.split() + ['EOS']
        qc_idx = torch.tensor(self.vocab.get_indices(qc_words)).to(self.device).unsqueeze(0)
        qc_len = torch.tensor(qc_idx.size()[1]).int().cpu().unsqueeze(0)
        # Whole sequence through encoder
        outputs = torch.zeros(100, 1, self.vocab.num_words).to(self.device)
        enc_hid = self.model.encoder.init_hidden(1, self.cfg['num_enc_layers']).to(self.device)
        enc_out, enc_hid = self.model.encoder(qc_idx, qc_len, enc_hid)
        dec_inp = torch.ones(1, device=self.device) * 1
        dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

        # One token at a time from decoder
        tok, i = 0, 0
        while tok != 2:  # EOS
            if self.model.decoder.att:
                dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid, enc_out=enc_out)
            else:
                dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
            try:
                outputs[i] = dec_out
            except IndexError:
                print('EOS token not found, but max length reached. Returning truncated sequence.')
                break
            tok = dec_out.argmax(1)
            dec_inp = tok
            i = i + 1
        pred_tok = torch.argmax(outputs.detach(), dim=2)
        pred_tok = torch.transpose(pred_tok, 1, 0)
        pred_sents = [' '.join(x) for x in self.vocab.get_sentence(pred_tok.cpu())][0]
        pred_sents = pred_sents.replace('PAD', '').strip()
        pred_sents = pred_sents[4:-4]
        print('out fr:', pred_sents)

    def test(self, test_loader, loss_fn, test_bsz):
        test_loss, test_bleu = self.valid_epoch(test_loader, loss_fn, test_bsz, write_outputs=True)
        print('Test loss: {0}'.format(test_loss))
        print('Test bleu: {0}'.format(test_bleu))
        self.test_losses.append(test_loss)
        self.test_bleu.append(test_bleu)

    def train(self, train_loader, valid_loader, loss_fn=None, lr=1e-2, weight_decay=1e-5, train_bsz=1, valid_bsz=1, num_epochs=1):
        enc_opt = torch.optim.Adam(self.model.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        dec_opt = torch.optim.Adam(self.model.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            max_valid_bleu = 0.0
            train_loss, train_bleu = self.train_epoch(train_loader, loss_fn, enc_opt, dec_opt, train_bsz)
            print('EPOCH {0} \t train_loss {1} \t train_bleu {2}'.format(epoch, train_loss, train_bleu))
            valid_loss, valid_bleu = self.valid_epoch(valid_loader, loss_fn, valid_bsz)
            print('\t valid_loss {1} \t valid_bleu {2}'.format(epoch, valid_loss, valid_bleu))
            # Early stop
            last_val_losses = self.valid_losses[-self.patience:]
            if epoch > self.patience:
                stop = True
                for l in last_val_losses:
                    if valid_loss < l:
                        stop = False
                        break
                if stop:
                    print('Early stopping: validation loss has not improved in {0} epochs.'.format(self.patience))
                    break
            if valid_bleu > max_valid_bleu:
                max_valid_bleu = valid_bleu
                self.save_model(os.path.join('models', self.name))
            # Log losses
            self.train_losses.append(train_loss)
            self.train_bleu.append(train_bleu)
            self.valid_losses.append(valid_loss)
            self.valid_bleu.append(valid_bleu)

    def train_epoch(self, train_loader, loss_fn, enc_opt, dec_opt, train_bsz=1):
        self.model.encoder.train()
        self.model.decoder.train()
        loss_epoch, bleu_epoch = 0.0, 0.0
        for i, (x, y, x_len, y_len) in enumerate(train_loader):
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            inp_qc = self.vocab.get_sentence(x)
            print('---')
            print('TRAIN inp qc', ' '.join(inp_qc[0]).replace('PAD', '').strip())
            inp_fr = self.vocab.get_sentence(y)
            print('TRAIN inp_fr', ' '.join(inp_fr[0]).replace('PAD', '').strip())

            x, y = x.to(self.device), y.to(self.device)
            enc_hid = self.model.encoder.init_hidden(train_bsz, self.cfg['num_enc_layers']).to(self.device)

            tgt_len = y.size(1)
            loss = 0.0

            # To store decoder outputs
            outputs = torch.zeros(tgt_len, train_bsz, self.vocab.num_words).to(self.device)

            # Whole sequence through encoder
            enc_out, enc_hid = self.model.encoder(x, x_len, enc_hid)

            # First input to the decoder is BOS (hardcoded: idx is 1)
            dec_inp = torch.ones(train_bsz, device=self.device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

            use_teach_forc = True if random.random() < self.model.decoder.teach_forc_ratio else False

            # One token at a time from decoder
            for di in range(tgt_len):  # Minus 2 so that we start at 0, and we exclude BOS
                if self.model.decoder.att:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid, enc_out=enc_out)
                else:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
                outputs[di] = dec_out
                tok = dec_out.argmax(1)
                # Teacher forcing: Feed the target as the next input
                if use_teach_forc:
                    dec_inp = y[:, di]
                else:
                    dec_inp = tok

            # Replace the predicted tokens that should be padding with padding
            mask = torch.arange(tgt_len).expand(len(y_len), tgt_len) < y_len.unsqueeze(1)
            mask = torch.transpose(mask, 1, 0).unsqueeze(2).float().to(self.device)
            outputs = outputs * mask

            # When calculating loss, collapse batches together
            all_outputs = outputs.view(-1, outputs.shape[-1])
            all_y = torch.transpose(y, 1, 0)
            all_y = all_y.reshape(-1)
            loss = loss_fn(all_outputs, all_y.long())

            loss.backward()
            pred_tok = torch.argmax(outputs.detach(), dim=2)
            pred_tok = torch.transpose(pred_tok, 1, 0)
            pred_sents = [' '.join(x) for x in self.vocab.get_sentence(pred_tok.cpu())]
            print('TRAIN out_fr', pred_sents[0].replace('PAD', '').strip())

            enc_opt.step()
            dec_opt.step()

            # Report loss
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch

            # Store BLEU ngram counts
            self.bleu(pred_tok, y)

        # Calculate BLEU over everything seen in epoch
        bleu_epoch = self.bleu.get_metric(reset=True)['BLEU']
        return loss_epoch, bleu_epoch

    def valid_epoch(self, valid_loader, loss_fn, valid_bsz=1, write_outputs=False):
        self.model.encoder.eval()
        self.model.decoder.eval()
        loss_epoch, bleu_epoch = 0.0, 0.0
        for i, (x, y, x_len, y_len) in enumerate(valid_loader):
            if write_outputs: # Writes only some outputs
                write_idx = self.write_idx
                inp_qc = ' '.join(self.vocab.get_sentence(x)[write_idx])
                inp_fr = ' '.join(self.vocab.get_sentence(y)[write_idx])

            x, y = x.to(self.device), y.to(self.device)
            enc_hid = self.model.encoder.init_hidden(valid_bsz, self.cfg['num_enc_layers']).to(self.device)

            tgt_len = y.size(1)
            loss = 0.0

            # To store decoder outputs
            outputs = torch.zeros(tgt_len, valid_bsz, self.vocab.num_words).to(self.device)

            # Whole sequence through encoder
            enc_out, enc_hid = self.model.encoder(x, x_len, enc_hid)

            # First input to the decoder is BOS (hardcoded: idx is 1)
            dec_inp = torch.ones(valid_bsz, device=self.device) * 1
            dec_hid = enc_hid  # First decoder hidden state is last encoder hidden state

            # One token at a time from decoder
            for di in range(tgt_len - 1):
                if self.model.decoder.att:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid, enc_out=enc_out)
                else:
                    dec_out, dec_hid = self.model.decoder(dec_inp, dec_hid)
                outputs[di] = dec_out
                tok = dec_out.argmax(1)
                # No teacher forcing: next input is current output
                dec_inp = tok

            # Replace the predicted tokens that should be padding with padding
            mask = torch.arange(tgt_len).expand(len(y_len), tgt_len) < y_len.unsqueeze(1)
            mask = torch.transpose(mask, 1, 0).unsqueeze(2).float().to(self.device)
            outputs = outputs * mask

            # When calculating loss, collapse batches together
            all_outputs = outputs.view(-1, outputs.shape[-1])
            all_y = torch.transpose(y, 1, 0)
            all_y = all_y.reshape(-1)
            loss = loss_fn(all_outputs, all_y.long())

            # Report loss
            loss_batch = loss.item() / tgt_len
            loss_epoch += loss_batch

            # Store BLEU ngram counts
            pred_tok = torch.argmax(outputs.detach(), dim=2)
            pred_tok = torch.transpose(pred_tok, 1, 0)
            self.bleu(pred_tok, y)

            if write_outputs:
                pred_sents = [' '.join(x) for x in self.vocab.get_sentence(pred_tok.cpu())][write_idx]
                self.outputs.append('inp_qc: {0}'.format(inp_qc))
                self.outputs.append('inp_fr: {0}'.format(inp_fr))
                self.outputs.append('out_fr: {0}'.format(pred_sents))
                self.outputs.append('----')

        # Calculate BLEU over everything seen in epoch
        bleu_epoch = self.bleu.get_metric(reset=True)['BLEU']
        return loss_epoch, bleu_epoch


class Seq2SeqArch(nn.Module):
    def __init__(self, vocab, cfg, device, bi=True, att=False, batch_norm=True, teach_forc_ratio=0.5, dropout=0.0):
        '''
        This is a helper class that itself does nothing,
        but putting all the model parts together here facilitates
        saving/loading weights in just one model file.
        '''
        super(Seq2SeqArch, self).__init__()
        self.encoder = Encoder(vocab.num_words, cfg['embedding_size'],
                               cfg['hidden_size'], cfg['num_enc_layers'], device,
                               bi=bi, batch_norm=batch_norm, dropout=dropout).to(device)
        self.decoder = Decoder(cfg['hidden_size'], cfg['embedding_size'],
                               vocab.num_words, cfg['num_dec_layers'], device,
                               att=att,
                               teach_forc_ratio=teach_forc_ratio).to(device)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, device, bi=True, batch_norm=True, dropout=0.0):
        super(Encoder, self).__init__()
        self.device = device
        self.bi = bi
        self.batchnorm = batch_norm
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bi)
        self.dropout = torch.nn.Dropout(p=dropout)
        if batch_norm:
           self.batchnorm = torch.nn.BatchNorm1d(embedding_size)
        if bi:
            self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, x_lens, hidden):
        x = x.long()
        embedded = self.embedding(x)
        
        if self.batchnorm:
            embedded = self.batchnorm(embedded.permute(0, 2, 1))
            embedded = embedded.permute(0, 2, 1)
        
        # Ignore the padding through the RNN
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens,
                                                           batch_first=True,
                                                        enforce_sorted=False)
        output, hidden = self.gru(embedded, hidden)

        # Re-pad
        output, output_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        if self.bi:
            # Combine forward and backward RNN states
            hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            hidden = torch.tanh(hidden).unsqueeze(0)
        return output, hidden

    def init_hidden(self, batch_size, num_layers):
        dim = 2 if self.bi else 1
        return torch.zeros(dim * num_layers, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, num_layers, device, att=False, max_length=100, teach_forc_ratio=0.5):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.teach_forc_ratio = teach_forc_ratio
        self.att = att
        if att:
            self.attn_l1 = nn.Linear((hidden_size * 3), hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
            self.gru = nn.GRU((hidden_size * 2) + embedding_size, hidden_size)
            self.out = nn.Linear((hidden_size * 3) + embedding_size, output_size)

    def forward(self, x, hidden, enc_out=None):
        # Input here is always one token at a time,
        # so need to do some unsqueezing to account for length dimension (1)
        x = x.long().unsqueeze(0)
        embedded = self.embedding(x)
        if self.att:
            bsz = enc_out.shape[0]
            src_len = enc_out.shape[1]
            # Repeat hidden state for every timestep (makes dims match)
            hid_rep = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
            concat = torch.cat((hid_rep, enc_out), 2)
            att_l1 = self.attn_l1(concat).permute(0, 2, 1)
            # Repeat for every example in the batch
            v = self.v.repeat(bsz, 1).unsqueeze(1)
            attn = torch.bmm(v, att_l1).squeeze(1)
            attn = F.softmax(attn, dim=1)
            attn = attn.unsqueeze(1)
            # Weight the encoder states
            context = torch.bmm(attn, enc_out).permute(1, 0, 2)
            to_gru = torch.cat((embedded, context), 2)
        else:
            to_gru = embedded
        to_gru = F.relu(to_gru)
        output, hidden = self.gru(to_gru, hidden)
        if self.att:
            to_out = torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), 1)
            output = self.out(to_out)
            return output, hidden  #, attn_weights
        else:
            output = self.out(output.squeeze(0))
            return output, hidden
