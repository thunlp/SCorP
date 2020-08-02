import json
import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sememe_number = 1400
data_path = 'data'
train_data_path = os.path.join(data_path, 'train_data.json')
valid_data_path = os.path.join(data_path, 'valid_data.json')
test_data_path = os.path.join(data_path, 'test_data.json')
word_index_path = os.path.join(data_path, 'word2index.json')
index_word_path = os.path.join(data_path, 'index2word.json')
word_vec_path = os.path.join(data_path, 'word_vector.npy')
index_sememe_path = os.path.join(data_path, 'index_sememe.json')
mp = True
gw = True
se = True
batch_size = 128
hidden_size = 256
lstm_layers = 1
epoch_num = 100
device = torch.device('cuda')


class BiSentenceLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers == 1:
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)

    def forward(self, x, x_len):
        # x: Tensor(batch, length, input_size) float32
        # x_len: Tensor(batch) int64
        # x_len_sort_idx: Tensor(batch) int64
        _, x_len_sort_idx = torch.sort(-x_len)
        # x_len_sort_idx: Tensor(batch) int64
        _, x_len_unsort_idx = torch.sort(x_len_sort_idx)
        x = x[x_len_sort_idx]
        x_len = x_len[x_len_sort_idx]
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        # ht: Tensor(num_layers * 2, batch, hidden_size) float32
        # ct: Tensor(num_layers * 2, batch, hidden_size) float32
        h_packed, (ht, ct) = self.lstm(x_packed, None)
        ht = ht[:, x_len_unsort_idx, :]
        ct = ct[:, x_len_unsort_idx, :]
        # h: Tensor(batch, length, hidden_size * 2) float32
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        h = h[x_len_unsort_idx]
        return h, (ht, ct)


class SememeEmbedding(torch.nn.Module):
    def __init__(self, sememe_number, embedding_dim):
        super().__init__()
        self.sememe_number = sememe_number
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(self.sememe_number + 1, self.embedding_dim, padding_idx=self.sememe_number, max_norm=5, sparse=True)
        self.embedding.weight.data[self.sememe_number] = 0

    def forward(self, x):
        # x: T(batch_size, max_word_number, max_sememe_number) padding: self.sememe_number
        # x_mask: T(batch_size, max_word_number, max_sememe_number)
        x_mask = torch.lt(x, self.sememe_number).to(torch.float32)
        # x_embedding: T(batch_size, max_word_number, max_sememe_number, embedding_dim)
        x_embedding = self.embedding(x)
        # x_average: T(batch_size, max_word_number, embedding_dim)
        x_average = torch.sum(x_embedding, dim=2) / torch.max(torch.sum(x_mask, dim=2, keepdim=True), torch.tensor([[[1]]], device=device, dtype=torch.float32))
        return x_average


class SPLSTM(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, train_word2sememe):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.sememe_number = sememe_number
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.mp = mp
        self.gw = gw
        self.se = se
        self.embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False
        self.sememe_embedding = SememeEmbedding(self.sememe_number, self.embedding_dim)
        self.embedding_dropout = torch.nn.Dropout()
        self.lstmencoder = BiSentenceLSTM(self.embedding_dim, self.hidden_size, self.lstm_layers)
        self.fc = torch.nn.Linear(self.hidden_size * 2, self.sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        self.train_word2sememe = train_word2sememe

    def forward(self, operation, x=None, y=None):
        # x: T(batch_size, max_word_number) 0表示占位符
        # y: T(batch_size, sememe_number) 输入每个样例对应的sememe序号. 用-1表示已经输入完毕
        # x_word_embedding: T(batch_size, max_word_number, embedding_dim)
        x_word_embedding = self.embedding(x)
        if self.se:
            # x_sememe: T(batch_size, max_word_number, max_sememe_number)
            x_sememe = self.train_word2sememe[x]
            if self.gw:
                x_sememe[:, 0, :] = self.sememe_number
            # x_sememe_embedding: T(batch_size, max_word_number, embedding_dim)
            x_sememe_embedding = self.sememe_embedding(x_sememe)
            # x_embedding: T(batch_size, max_word_number, max_sememe_number)
            x_embedding = x_word_embedding + x_sememe_embedding
        else:
            x_embedding = x_word_embedding
        x_embedding = self.embedding_dropout(x_embedding)
        # mask: T(batch_size, max_word_number)
        mask = torch.gt(x, 0).to(torch.int64)
        # x_len: T(batch_size)
        x_len = torch.sum(mask, dim=1)
        if self.mp:
            # h: T(batch_size, max_word_number, hidden_size * 2)
            h, _ = self.lstmencoder(x_embedding, x_len)
            # pos_score: T(batch_size, max_word_number, sememe_number)
            pos_score = self.fc(h)
            mask_3 = mask.to(torch.float32).unsqueeze(2)
            pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
            # score: T(batch_size, sememe_number)
            score, _ = torch.max(pos_score, dim=1)
        else:
            # h: T(lstm_layers * 2, batch_size, hidden_size)
            _, (h, _) = self.lstmencoder(x_embedding, x_len)
            # h: T(batch_size, hidden_size * 2)
            h = torch.transpose(h[h.shape[0] - 2:, :, :], 0, 1).contiguous().view(x_len.shape[0], self.hidden_size * 2)
            # score: T(batch_size, sememe_number)
            score = self.fc(h)
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, y)
            return loss, score, indices
        elif operation == 'inference':
            return score, indices


def write_statistics(writer, name, tensor, step):
    def write_mean_std_max_min_absmean(writer, name, tensor, step):
        if tensor.shape[0] == 0:
            return
        name = 'zz.' + name
        writer.add_scalar(name + 'mean', torch.mean(tensor).item(), global_step=step)
        writer.add_scalar(name + 'std', torch.std(tensor).item(), global_step=step)
        writer.add_scalar(name + 'max', torch.max(tensor).item(), global_step=step)
        writer.add_scalar(name + 'min', torch.min(tensor).item(), global_step=step)
        writer.add_scalar(name + 'absmean', torch.mean(torch.abs(tensor)).item(), global_step=step)
        writer.add_histogram(name + 'histogram', tensor.detach().cpu().data.numpy(), global_step=step)
    if tensor.dtype not in [torch.float, torch.double, torch.half]:
        return
    if tensor.is_sparse:
        write_mean_std_max_min_absmean(writer, name + '/', tensor._values(), step)
    else:
        write_mean_std_max_min_absmean(writer, name + '/', tensor, step)
    if tensor.grad is not None:
        if tensor.grad.is_sparse:
            write_mean_std_max_min_absmean(writer, name + '/grad-', tensor.grad._values(), step)
        else:
            write_mean_std_max_min_absmean(writer, name + '/grad-', tensor.grad, step)


def evaluate(ground_truth, prediction):
    index = 1
    correct = 0
    point = 0
    for predicted_sememe in prediction:
        if predicted_sememe in ground_truth:
            correct += 1
            point += (correct / index)
        index += 1
    point /= len(ground_truth)
    return point


def build_word2sememe(train_data, word_number):
    max_sememe_number = max([len(instance['sememe_idx']) for instance in train_data])
    r = np.zeros((word_number, max_sememe_number), dtype=np.int64)
    r.fill(sememe_number)
    for instance in train_data:
        word_idx = instance['word_idx']
        sememes = instance['sememe_idx']
        r[word_idx, 0:len(sememes)] = np.array(sememes)
    r = torch.tensor(r, dtype=torch.int64, device=device)
    return r


def build_sentence_numpy(sentences):
    max_length = max([len(sentence) for sentence in sentences])
    sentence_numpy = np.zeros((len(sentences), max_length), dtype=np.int64)
    for i in range(len(sentences)):
        sentence_numpy[i, 0:len(sentences[i])] = np.array(sentences[i])
    return sentence_numpy


def get_sememe_label(sememes):
    l = np.zeros((len(sememes), sememe_number), dtype=np.float32)
    for i in range(len(sememes)):
        for s in sememes[i]:
            l[i, s] = 1
    return l
            

def sp_collate_fn(batch):
    words = [instance['word_idx'] for instance in batch]
    sememes = [instance['sememe_idx'] for instance in batch]
    if gw:
        definition_words = [instance['extended_definition_idx'] for instance in batch]
    else:
        definition_words = [instance['definition_idx'] for instance in batch]
    words_t = torch.tensor(np.array(words), dtype=torch.int64, device=device)
    sememes_t = torch.tensor(get_sememe_label(sememes), dtype=torch.float32, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    return words_t, sememes_t, definition_words_t, sememes


def get_dataloader(train_data, valid_data, test_data):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader


def load_map_data():
    word2index = json.load(open(word_index_path))
    index2word = json.load(open(index_word_path))
    word2vec = np.load(word_vec_path)
    index_sememe = json.load(open(index_sememe_path))
    return word2index, index2word, word2vec, index_sememe


def load_data():
    train_data = json.load(open(train_data_path))
    valid_data = json.load(open(valid_data_path))
    test_data = json.load(open(test_data_path))
    return train_data, valid_data, test_data


def lstm_train():
    word2index, index2word, word2vec, index_sememe = load_map_data()
    train_data, valid_data, test_data = load_data()
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(train_data, valid_data, test_data)
    writer = SummaryWriter('output')
    valid_words = [index2word[instance['word_idx']] for instance in valid_data]
    valid_words.sort()
    valid_words = ' '.join(valid_words)
    writer.add_text('valid_words', valid_words, global_step=0)
    train_word2sememe = build_word2sememe(train_data, len(word2index))
    model = SPLSTM(vocabulary_size=len(word2index), embedding_dim=word2vec.shape[1], train_word2sememe=train_word2sememe)
    model.embedding.weight.data = torch.from_numpy(word2vec)
    model.to(device)
    sparse_parameters_name = ['embedding.weight', 'sememe_embedding.embedding.weight']
    sparse_parameters = [para for name, para in model.named_parameters() if name in sparse_parameters_name]
    non_sparse_parameters = [para for name, para in model.named_parameters() if name not in sparse_parameters_name]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, non_sparse_parameters), lr=0.001)
    sparse_optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, sparse_parameters), lr=0.001)
    max_valid_map = 0
    max_valid_epoch = 0
    for epoch in range(epoch_num):
        print('epoch', epoch)
        model.train()
        train_map = 0
        train_loss = 0
        for words_t, sememes_t, definition_words_t, sememes in tqdm(train_dataloader):
            optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            sparse_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            for i in range(len(sememes)):
                train_map += evaluate(sememes[i], predicted[i])
            train_loss += loss.item()
        writer.add_scalar('train/loss', train_loss / len(train_data), epoch)
        writer.add_scalar('train/map', train_map / len(train_data), epoch)
        for name, parameter in model.named_parameters():
            write_statistics(writer, name, parameter, epoch)
        model.eval()
        valid_map = 0
        valid_loss = 0
        for words_t, sememes_t, definition_words_t, sememes in tqdm(valid_dataloader):
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t)
            words = words_t.detach().cpu().numpy().tolist()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                word = index2word[words[i]]
                word_str = 'valid_result/' + word
                m = evaluate(sememes[i], predicted[i])
                writer.add_text(word_str + '/map', str(m), global_step=epoch)
                valid_map += m
                writer.add_text(word_str + '/answer', ' '.join([index_sememe[sememe] for sememe in sememes[i]]), global_step=epoch)
                writer.add_text(word_str + '/prediction', '\n\n'.join([index_sememe[predicted[i][j]] + ' ' + str(score[i, predicted[i][j]]) for j in range(10)]), global_step=epoch)
            valid_loss += loss.item()
        writer.add_scalar('valid/loss', valid_loss / len(valid_data), epoch)
        writer.add_scalar('valid/map', valid_map / len(valid_data), epoch)
        print(f'train loss {train_loss / len(train_data)}, train map {train_map / len(train_data)}, valid loss {valid_loss / len(valid_data)}, valid map {valid_map / len(valid_data)}')
        if valid_map / len(valid_data) > max_valid_map:
            max_valid_epoch = epoch
            max_valid_map = valid_map / len(valid_data)
            torch.save(model.state_dict(), os.path.join('output', 'model'))
    model.load_state_dict(torch.load(os.path.join('output', 'model')))
    test_map = 0
    test_loss = 0
    predictions = dict()
    for words_t, sememes_t, definition_words_t, sememes in tqdm(test_dataloader):
        loss, score, indices = model('train', x=definition_words_t, y=sememes_t)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        words = words_t.detach().cpu().numpy().tolist()
        for i in range(len(sememes)):
            test_map += evaluate(sememes[i], predicted[i])
            predictions[words[i]] = [[sememe, s] for sememe, s in enumerate(score[i])]
        test_loss += loss.item()
    writer.add_scalar('test/loss', test_loss / len(test_data), epoch)
    writer.add_scalar('test/map', test_map / len(test_data), epoch)
    print(f'test loss {test_loss / len(test_data)}, test map {test_map / len(test_data)}')
    json.dump(predictions, open(os.path.join('output', 'prediction.json'), 'w'))
    json.dump(test_map / len(test_data), open(os.path.join('output', 'map.json'), 'w'))


if __name__ == '__main__':
    lstm_train()
