import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim (32, 20, 32)
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim (20, 32, 32)
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim # 여기서 hidden 은 init state. (4, 32, 32)
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim (32, 4, 32)
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim # (32, 32)
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out) # 확률값 뽑힘.
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0]) # (4, 32, 32)
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

if __name__ == "__main__":
    VOCAB_SIZE = 5000
    MAX_SEQ_LEN = 20
    START_LETTER = 0
    BATCH_SIZE = 32
    MLE_TRAIN_EPOCHS = 100
    ADV_TRAIN_EPOCHS = 50
    POS_NEG_SAMPLES = 10000

    GEN_EMBEDDING_DIM = 32
    GEN_HIDDEN_DIM = 32
    DIS_EMBEDDING_DIM = 64
    DIS_HIDDEN_DIM = 64
    CUDA = False
    gen = Discriminator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    inp = torch.zeros((32, 20), dtype=torch.long)
    target = torch.zeros((32, 20), dtype=torch.long)

    gen.batchClassify(inp)