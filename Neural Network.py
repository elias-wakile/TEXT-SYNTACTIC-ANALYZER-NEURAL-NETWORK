import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------
ACC_Y_LABEL = "accuracy"
LOSS_Y_LABEL = "loss"
ACC_BASED_ON_EPOCH = "accuracy based on epoch"
TRAIN_BASED_ON_EPOCH = "loss based on epoch"
X_LABEL = "epochs"

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

BATCH_SIZE = 64


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot(y_axis, x_label, y_label, title, legend, name):
    """
    plots train and validation metric and saves as image
    :param y_axis: values for y axis
    :param x_label: label on the x axis
    :param y_label: label on the y axis
    :param title: title of plot
    :param legend: legend for items
    :param name: name of image
    """
    plt.figure()
    x_axis = np.arange(1, y_axis.shape[1] + 1)
    plt.plot(x_axis, y_axis[0], label=legend[0])
    plt.plot(x_axis, y_axis[1], label=legend[1])
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{name}.png")


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    # caused crashes to the code so commented
    # vocab = list(wv_from_bin.vocab.keys())
    # print(wv_from_bin.vocab[vocab[0]])
    # print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    vec = np.zeros(embedding_dim)
    count = 0
    for word in sent.text:
        if word in word_to_vec:
            vec += word_to_vec[word]
            count += 1

    return vec / (count if count != 0 else 1)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    ind = [word_to_ind[word] for word in sent.text]
    return get_one_hot(len(word_to_ind), ind) / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {words_list[i]: i for i in range(len(words_list))}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """

    embedding_mat = np.zeros((seq_len, embedding_dim))
    for i, word in zip(range(seq_len), sent.text):
        if word in word_to_vec:
            embedding_mat[i] = word_to_vec[word]
    return embedding_mat


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path,
                                                               split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[
                TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(
                                         words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list, True),
                "embedding_dim": embedding_dim
            }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for
            k, sentences in self.sentences.items()}
        self.torch_iterators = {
            k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
            for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array(
            [sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape

    def get_dataset(self, data_subset=TRAIN):
        return self.torch_datasets[data_subset]


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                               batch_first=True, dropout=dropout,
                               bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        out, (h, c) = self.bi_lstm(text)
        nor_out = out[:, -1, :self.hidden_size]
        rev_out = out[:, 0, self.hidden_size:]
        return self.linear(torch.cat([nor_out, rev_out], dim=1))

    def predict(self, text):
        return torch.round(torch.sigmoid(self.forward(text)))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.round(torch.sigmoid(self.linear(x)))


# ------------------------- training functions -------------
def model_normal_eval(acc, loss, manager, model, loss_img_name, acc_img_name):
    """
    evaluates the mode on basic metric of loss,accuracy on train,validation
    and test
    :param acc: accuracy array for train and validation
    :param loss: loss array for train and validation
    :param manager: data manager
    :param model: model to evaluate
    :param loss_img_name: loss image to save
    :param acc_img_name: accuracy image to save
    """
    plot(loss, X_LABEL, LOSS_Y_LABEL, TRAIN_BASED_ON_EPOCH, ["train loss",
                                                             "val loss"],
         loss_img_name)
    plot(acc, X_LABEL, ACC_Y_LABEL, ACC_BASED_ON_EPOCH,
         ["train accuracy",
          "val accuracy"],
         acc_img_name)
    loss_test, acc_test = evaluate(model, manager.get_torch_iterator(TEST),
                                   nn.BCEWithLogitsLoss())
    print(f"last loss on train:{loss[0, -1]}")
    print(f"last loss on validation:{loss[1, -1]}")
    print(f"last accuracy on train:{acc[0, -1]}")
    print(f"last accuracy on validation:{acc[1, -1]}")
    print(f"loss on test:{loss_test}")
    print(f"acc on test:{acc_test}")


def model_special_eval(manager, model):
    """
    evaluates the model on the special sub datasets
    :param manager: data manager
    :param model: model to evaluate
    """
    negated_loader, rare_loader = get_special_loaders(manager, BATCH_SIZE)
    loss_negated, acc_negated = evaluate(model, negated_loader,
                                         nn.BCEWithLogitsLoss())
    loss_rare, acc_rare = evaluate(model, rare_loader, nn.BCEWithLogitsLoss())
    print(f"loss on negated set:{loss_negated}")
    print(f"acc on negated set:{acc_negated}")
    print(f"loss on set with rare words:{loss_rare}")
    print(f"acc on set with rare words:{acc_rare}")


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return torch.sum(torch.eq(torch.round(preds), y)).item() / y.size()[0]


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    avg_loss = 0
    avg_acc = 0
    for embed, labels in data_iterator:
        # print(torch.any(torch.isnan(embed)))
        output = model(embed.float())
        labels = labels.reshape(-1, 1).float()
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        with torch.no_grad():
            avg_acc += binary_accuracy(torch.sigmoid(output), labels)

    return avg_loss / len(data_iterator), avg_acc / len(data_iterator)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    avg_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for embed, label in data_iterator:
            output = model(embed.float())
            labels = label.reshape(-1, 1).float()
            loss = criterion(output, labels)
            avg_loss += loss.item()
            avg_acc += binary_accuracy(torch.sigmoid(output), labels)
    return avg_loss / len(data_iterator), avg_acc / len(data_iterator)


def get_special_loaders(data_manager, batch_size):
    """
    creates data loaders for the special subset of data
    :param data_manager: data manager to use
    :param batch_size: batch size
    :return: data loader for negated sent and data loader for sentences with
    rare words
    """
    test_dataset = data_manager.get_dataset(TEST)
    sentences = data_manager.sentences[TEST]
    negated = data_loader.get_negated_polarity_examples(sentences)
    rare = data_loader.get_rare_words_examples(sentences,
                                               data_manager.sentiment_dataset)
    negated_dataset = torch.utils.data.Subset(test_dataset, negated)
    rare_dateset = torch.utils.data.Subset(test_dataset, rare)
    negated_loader = torch.utils.data.DataLoader(negated_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    rare_loader = torch.utils.data.DataLoader(rare_dateset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return negated_loader, rare_loader


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """

    return torch.cat(
            [model.predict(embeds) for embeds, labels in data_iter])


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loss_values = np.zeros((2, n_epochs))
    acc_values = np.zeros((2, n_epochs))
    train_loader = data_manager.get_torch_iterator(TRAIN)
    val_loader = data_manager.get_torch_iterator(VAL)
    for i in range(n_epochs):
        print(i)
        loss_t, acc_t = train_epoch(model,
                                    train_loader,
                                    optimizer,
                                    criterion)
        print(loss_t, " ", acc_t)
        loss_v, acc_v = evaluate(model,
                                 val_loader,
                                 criterion)
        print(loss_v, " ", acc_v)

        loss_values[0, i], loss_values[1, i] = loss_t, loss_v
        acc_values[0, i], acc_values[1, i] = acc_t, acc_v
    return loss_values, acc_values


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """

    manager = DataManager(ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    model = LogLinear(manager.get_input_shape()[0])
    loss, acc = train_model(model, manager, 20, 0.01, 0.001)
    model_normal_eval(acc, loss, manager, model, "loss_epoch_log_linear",
                      "accuracy_epoch_log_linear")
    model_special_eval(manager, model)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model
    with word embeddings representation.
    """
    manager = DataManager(W2V_AVERAGE, batch_size=BATCH_SIZE,
                          embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(manager.get_input_shape()[0])
    loss, acc = train_model(model, manager, 20, 0.01, 0.001)
    model_normal_eval(acc, loss, manager, model, "loss_epoch_log_linear_w2v",
                      "accuracy_epoch_log_linear_w2v")
    model_special_eval(manager, model)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    manager = DataManager(W2V_SEQUENCE, batch_size=BATCH_SIZE,
                          embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(W2V_EMBEDDING_DIM, 100, 1, 0)
    loss, acc = train_model(model, manager, 4, 0.001, 0.0001)
    model_normal_eval(acc, loss, manager, model, "loss_epoch_lstm",
                      "accuracy_epoch_lstm")
    model_special_eval(manager, model)


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()
    # manager = DataManager(W2V_SEQUENCE, batch_size=BATCH_SIZE,
    #                        embedding_dim=W2V_EMBEDDING_DIM)
    # negated_loader, rare_loader = get_special_loaders(manager, BATCH_SIZE)
    # print()
    pass
