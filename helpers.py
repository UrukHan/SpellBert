import os
import pickle5 as pickle #import pickle
import sys
from typing import List

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

BERT_MAX_SEQ_LEN = 512
BERT_TOKENIZER = None


def progressBar(value, endvalue, names, values, bar_length=30):
    assert (len(names) == len(values))
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = ''

    for name, val in zip(names, values):
        temp = '|| {0}: {1:.2f} '.format(name, val) if val is not None else '|| {0}: {1} '.format(name, None)
        string += temp
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    return

#----------

def token_correct(token):
    temp = list(""",;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{}""")
    out = ''
    for i in range(len(token)):
        if i != len(token) - 1:
            if token[i] in temp:
                pass
            else:
                out += token[i]
        else:
            out += token[i]
    return out
 
def like_tokens(a, b):
    if max(len(set(a).difference(set(b))), len(set(b).difference(set(a)))) <= 3:
        return True
    else:
        return False

def phrase_correct(correct, corrupt):

    temp = list(""",;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{}""")
    spec = 'ъъъ'

    for i in range(len(corrupt)):
        try:
            if i != 0 and len(corrupt)-1 > i:
                if corrupt[i][0] in temp and correct[i][0] not in temp and len(corrupt[i]) > 1:
                    corrupt = corrupt[:i] + [corrupt[i][0]] + [corrupt[i][1:]] + corrupt[i+1:]
                    
            if i <= len(corrupt)-1 and i <= len(correct)-1:
                if like_tokens(corrupt[i], correct[i]):
                    pass
                elif i+2 <= len(corrupt)-1 and i+2 <= len(correct)-1:
                    if like_tokens(corrupt[i + 1], correct[i + 1]) or \
                            like_tokens(corrupt[i + 2], correct[i + 2]):
                        pass
                    elif (like_tokens(corrupt[i + 2], correct[i + 1]) and not \
                                like_tokens(corrupt[i + 2], correct[i + 2])) or\
                                    (like_tokens(corrupt[i + 2], correct[i]) and not \
                                        like_tokens(corrupt[i + 2], correct[i + 2])and not \
                                            like_tokens(corrupt[i + 2], correct[i + 1])):
                        correct = correct[:i] + [spec] + correct[i:]

                    elif len(corrupt) >= len(correct):
                        correct = correct[:i] + [spec] + correct[i:]
            
                    else:
                        corrupt = corrupt[:i] + [correct[i]] + corrupt[i:]

                elif i+1 <= len(corrupt)-1 and i+1 <= len(correct)-1:
                    if like_tokens(corrupt[i + 1], correct[i + 1]):
                        pass
                    elif len(corrupt) >= len(correct):
                        correct = correct[:i] + [spec, correct[i]] + correct[i+1:]
                        
                    else:
                        corrupt = corrupt[:i] + [correct[i]] + corrupt[i:]
                else:
                    pass

        

            elif like_tokens(corrupt[i], correct[-1]):
                correct = correct[:-1] + [spec] + [correct[-1]]
            else:
                correct = correct[:i] + [spec] + correct[i:] 

            if correct[i][-1] in temp or corrupt[i][-1] in temp:            
                if correct[i][-1] == corrupt[i][-1]:
                    pass
                elif corrupt[i][-1] in temp and correct[i][-1] not in temp:
                    corrupt[i] = corrupt[i][:-1]
                else:
                    corrupt[i] = corrupt[i] + correct[i][-1]
        except IndexError:
            pass

    return correct, corrupt

#---------

def load_data(base_path, corr_file, incorr_file):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r", encoding="utf8")
    for line in opfile1:
        if line.strip() != "": incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r", encoding="utf8")
    for line in opfile2:
        if line.strip() != "": corr_data.append(line.strip())
    opfile2.close()
    assert len(incorr_data) == len(corr_data)

    # verify if token split is same
    for i, (x, y) in tqdm(enumerate(zip(corr_data, incorr_data))):
        x_split, y_split = list(map(token_correct, x.split())), list(map(token_correct, y.split()))
        x_split, y_split = phrase_correct(x_split, y_split)

        try:
            corr_data[i] = " ".join(x_split)
            incorr_data[i] = " ".join(y_split)
            assert len(x_split) == len(y_split)
            
        except AssertionError:
            #print("# tokens in corr and incorr mismatch. retaining and trimming to min len.")
            #print(x_split)
            #print(y_split)
            mn = min([len(x_split), len(y_split)])
            corr_data[i] = " ".join(x_split[:mn])
            incorr_data[i] = " ".join(y_split[:mn])
            #print(corr_data[i])
            #print(incorr_data[i])

    # return as pairs
    data = []
    for x, y in tqdm(zip(corr_data, incorr_data)):
        data.append((x, y))

    print(f"loaded tuples of (corr,incorr) examples from {base_path}")
    return data


def train_validation_split(data, train_ratio, seed):
    np.random.seed(seed)
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio * len_))
    inds_shuffled = np.arange(len_)#;
    np.random.shuffle(inds_shuffled)#;
    train_data = []
    for ind in inds_shuffled[:train_len_]: train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]: validation_data.append(data[ind])
    return train_data, validation_data


def get_char_tokens(use_default: bool, data = None):
    if not use_default and data is None: raise Exception("data is None")

    # reset char token utils
    chartoken2idx, idx2chartoken = {}, {}
    char_unk_token, char_pad_token, char_start_token, char_end_token = \
        "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>"
    special_tokens = [char_unk_token, char_pad_token, char_start_token, char_end_token]
    for char in special_tokens:
        idx = len(chartoken2idx)
        chartoken2idx[char] = idx
        idx2chartoken[idx] = char

    if use_default:
        chars = len(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))
        for char in chars:
            if char not in chartoken2idx:
                idx = len(chartoken2idx)
                chartoken2idx[char] = idx
                idx2chartoken[idx] = char
    else:
        # realized that set doesn't preserve order!!
        for line in tqdm(data):
            for char in line:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char

    print(f"number of unique chars found: {len(chartoken2idx)}")
    print(chartoken2idx)
    return_dict = {}
    return_dict["chartoken2idx"] = chartoken2idx
    return_dict["idx2chartoken"] = idx2chartoken
    return_dict["char_unk_token"] = char_unk_token
    return_dict["char_pad_token"] = char_pad_token
    return_dict["char_start_token"] = char_start_token
    return_dict["char_end_token"] = char_end_token
    # new
    return_dict["char_unk_token_idx"] = chartoken2idx[char_unk_token]
    return_dict["char_pad_token_idx"] = chartoken2idx[char_pad_token]
    return_dict["char_start_token_idx"] = chartoken2idx[char_start_token]
    return_dict["char_end_token_idx"] = chartoken2idx[char_end_token]

    return return_dict


def get_tokens(data,
               keep_simple=False,
               min_max_freq=(1, float("inf")),
               topk=None,
               intersect=[],
               load_char_tokens=False):
    # get all tokens
    token_freq, token2idx, idx2token = {}, {}, {}
    for example in tqdm(data):
        for token in example.split():
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
    print(f"Total tokens found: {len(token_freq)}")

    # retain only simple tokens
    if keep_simple:
        isascii = lambda s: len(s) == len(s.encode()) or len(s) * 2 == len(s.encode())
        hasdigits = lambda s: len([x for x in list(s) if x.isdigit()]) > 0
        tf = [(t, f) for t, f in [*token_freq.items()] if (isascii(t) and not hasdigits(t))]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only tokens with specified min and max range
    if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        tf = [(i[0], i[1]) for i in sorted_ if (i[1] >= min_max_freq[0] and i[1] <= min_max_freq[1])]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only topk tokens
    if topk is not None:
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse = True)
        token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only interection of tokens
    if len(intersect) > 0:
        tf = [(t, f) for t, f in [*token_freq.items()] if (t in intersect or t.lower() in intersect)]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # create token2idx and idx2token
    for token in token_freq:
        idx = len(token2idx)
        idx2token[idx] = token
        token2idx[token] = idx

    # add <<PAD>> special token
    ntokens = len(token2idx)
    pad_token = "<<PAD>>"
    token_freq.update({pad_token: -1})
    token2idx.update({pad_token: ntokens})
    idx2token.update({ntokens: pad_token})

    # add <<UNK>> special token
    ntokens = len(token2idx)
    unk_token = "<<UNK>>"
    token_freq.update({unk_token: -1})
    token2idx.update({unk_token: ntokens})
    idx2token.update({ntokens: unk_token})

    # new
    # add <<EOS>> special token
    ntokens = len(token2idx)
    eos_token = "<<EOS>>"
    token_freq.update({eos_token: -1})
    token2idx.update({eos_token: ntokens})
    idx2token.update({ntokens: eos_token})

    # return dict
    token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
    return_dict = {"token2idx": token2idx,
                   "idx2token": idx2token,
                   "token_freq": token_freq,
                   "pad_token": pad_token,
                   "unk_token": unk_token,
                   "eos_token": eos_token
                   }
    # new
    return_dict.update({
        "pad_token_idx": token2idx[pad_token],
        "unk_token_idx": token2idx[unk_token],
        "eos_token_idx": token2idx[eos_token],
    })

    # load_char_tokens
    if load_char_tokens:
        print("loading character tokens")
        char_return_dict = get_char_tokens(use_default=False, data=data)
        return_dict.update(char_return_dict)

    return return_dict


# train utils

def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of lables and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:  np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch_labels = [data[idx][0] for idx in batch_indices]
        batch_sentences = [data[idx][1] for idx in batch_indices]
        yield (batch_labels, batch_sentences)


def labelize(batch_labels, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_labels]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def tokenize(batch_sentences, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_sentences]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff="pass-through"):
    assert backoff in ["neutral", "pass-through"], print(f"selected backoff strategy not implemented: {backoff}")
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]
    if backoff == "pass-through":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else clean_[i] for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    elif backoff == "neutral":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else "a" for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    return batch_predictions


def untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_clean_sentences, topk=None):
    """
    batch_predictions are softmax probabilities and should have shape (batch_size,max_seq_len,vocab_size)
    batch_lengths should have shape (batch_size)
    batch_clean_sentences should be strings of shape (batch_size)
    """
    # print(batch_predictions.shape)
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    if topk is not None:
        # get topk items from dim=2 i.e top 5 prob inds
        batch_predictions = np.argpartition(-batch_predictions, topk, axis=-1)[:, :,
                            :topk]  # (batch_size,max_seq_len,5)
    # else:
    #    batch_predictions = batch_predictions # already have the topk indices

    # get topk words
    idx_to_token = lambda idx, idx2token, corresponding_clean_token, unktoken: idx2token[
        idx] if idx != unktoken else corresponding_clean_token
    batch_predictions = \
        [[[idx_to_token(wordidx, idx2token, batch_clean_sentences[i][j], unktoken) \
           for wordidx in topk_wordidxs] \
          for j, topk_wordidxs in enumerate(predictions[:batch_lengths[i]])] \
         for i, predictions in enumerate(batch_predictions)]

    return batch_predictions


def get_model_nparams(model):
    ntotal = 0
    for param in list(model.parameters()):
        temp = 1
        for sz in list(param.size()): temp *= sz
        ntotal += temp
    return ntotal


def batch_accuracy_func(batch_predictions: np.ndarray,
                        batch_targets: np.ndarray,
                        batch_lengths: list):
    """
    given the predicted word idxs, this method computes the accuracy 
    by matching all values from 0 index to batch_lengths_ index along each 
    batch example
    """
    assert len(batch_predictions) == len(batch_targets) == len(batch_lengths)
    count_ = 0
    total_ = 0
    for pred, targ, len_ in zip(batch_predictions, batch_targets, batch_lengths):
        count_ += (pred[:len_] == targ[:len_]).sum()
        total_ += len_
    return count_, total_


def load_vocab_dict(path_: str):
    """
    path_: path where the vocab pickle file is saved
    """
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab


def save_vocab_dict(path_: str, vocab_: dict):
    """
    path_: path where the vocab pickle file to be saved
    vocab_: the dict data
    """
    with open(path_, 'wb') as fp:
        pickle.dump(vocab_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


################################################
# ----->
# For BERT Custom Tokenization
################################################


def merge_subtokens(tokens: List):
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"):
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    text = " ".join(merged_tokens)
    return text


def _custom_bert_tokenize_sentence(text):
    tokens = BERT_TOKENIZER.tokenize(text)
    tokens = tokens[:BERT_MAX_SEQ_LEN - 2]  # 2 allowed for [CLS] and [SEP]
    idxs = np.array([idx for idx, token in enumerate(tokens) if not token.startswith("##")] + [len(tokens)])
    split_sizes = (idxs[1:] - idxs[0:-1]).tolist()
    # NOTE: BERT tokenizer does more than just splitting at whitespace and tokenizing. So be careful.
    # -----> assert len(split_sizes)==len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes, text)
    # -----> hence do the following:
    text = merge_subtokens(tokens)
    assert len(split_sizes) == len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes,
                                                        text)
    return text, tokens, split_sizes


def _custom_bert_tokenize_sentences(list_of_texts):
    out = [_custom_bert_tokenize_sentence(text) for text in list_of_texts]
    texts, tokens, split_sizes = list(zip(*out))
    return [*texts], [*tokens], [*split_sizes]


def _simple_bert_tokenize_sentences(list_of_texts):
    return [merge_subtokens(BERT_TOKENIZER.tokenize(text)[:BERT_MAX_SEQ_LEN - 2]) for text in list_of_texts]


def bert_tokenize(batch_sentences):
    """
    inputs:
        batch_sentences: List[str]
            a list of textual sentences to tokenized
    outputs:
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    batch_sentences, batch_tokens, batch_splits = _custom_bert_tokenize_sentences(batch_sentences)

    # max_seq_len = max([len(tokens) for tokens in batch_tokens])
    # batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens,max_length=max_seq_len,pad_to_max_length=True) for tokens in batch_tokens]
    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    batch_attention_masks = pad_sequence(
        [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)
    batch_input_ids = pad_sequence([torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts],
                                   batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(
        [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)

    batch_bert_dict = {"attention_mask": batch_attention_masks,
                       "input_ids": batch_input_ids,
                       "token_type_ids": batch_token_type_ids}

    return batch_sentences, batch_bert_dict, batch_splits


def bert_tokenize_for_valid_examples(batch_orginal_sentences, batch_noisy_sentences, bert_pretrained_name_or_path=None):

    global BERT_TOKENIZER

    if BERT_TOKENIZER is None:  # gets initialized during the first call to this method
        if bert_pretrained_name_or_path:
            BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(bert_pretrained_name_or_path)
            BERT_TOKENIZER.do_basic_tokenize = True
            BERT_TOKENIZER.tokenize_chinese_chars = False
        else:
            BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational') #'bert-base-cased'
            BERT_TOKENIZER.do_basic_tokenize = True
            BERT_TOKENIZER.tokenize_chinese_chars = False

    _batch_orginal_sentences = _simple_bert_tokenize_sentences(batch_orginal_sentences)
    _batch_noisy_sentences, _batch_tokens, _batch_splits = _custom_bert_tokenize_sentences(batch_noisy_sentences)


    '''for idx, (a, b) in enumerate(zip(_batch_orginal_sentences, _batch_noisy_sentences)):
        if len(a.split()) != len(b.split()):
            print(idx)
            print(a.split())
            print(b.split())'''

    valid_idxs = [idx for idx, (a, b) in enumerate(zip(_batch_orginal_sentences, _batch_noisy_sentences)) if
                  len(a.split()) == len(b.split())]
    batch_orginal_sentences = [line for idx, line in enumerate(_batch_orginal_sentences) if idx in valid_idxs]
    batch_noisy_sentences = [line for idx, line in enumerate(_batch_noisy_sentences) if idx in valid_idxs]
    batch_tokens = [line for idx, line in enumerate(_batch_tokens) if idx in valid_idxs]
    batch_splits = [line for idx, line in enumerate(_batch_splits) if idx in valid_idxs]

    batch_bert_dict = {
        "attention_mask": [],
        "input_ids": [],
        # "token_type_ids": []
    }
    if len(valid_idxs) > 0:
        batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]
        batch_attention_masks = pad_sequence(
            [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_input_ids = pad_sequence(
            [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        # batch_token_type_ids = pad_sequence(
        #     [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        #     padding_value=0)
        batch_bert_dict = {"attention_mask": batch_attention_masks,
                           "input_ids": batch_input_ids,
                           # "token_type_ids": batch_token_type_ids
                           }

    return batch_orginal_sentences, batch_noisy_sentences, batch_bert_dict, batch_splits

################################################
# <-----
################################################
