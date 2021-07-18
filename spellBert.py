#%%

from corrector_subwordbert import BertChecker
from helpers import load_data, train_validation_split
from helpers import get_tokens
from preproc_text import load_clean, clean_txt, corrupt_txt
import os

#%%

data_dir = 'D:/Google/NeuroNet/BERT/SpellBert'
check_dir = os.path.join(data_dir, 'new_models')
clean_file = 'clean.txt'
corrupt_file = 'corrupt.txt'
path = os.path.join(data_dir, 'rt.csv.gz')


'''
%%

data = load_clean(path)
data[5]

%%

clean_txt(data)

%%

corrupt_txt(os.path.join(data_dir, clean_file))'''


#%%

train_data = load_data(data_dir, clean_file, corrupt_file)
train_data, test_data = train_validation_split(train_data, 0.9, seed = 11690)

# %%

vocab = get_tokens([i[0] for i in train_data], keep_simple = True, min_max_freq = (1, float("inf")), topk = 500000)

# %%

checker = BertChecker(device = "cuda") 
checker.from_huggingface(bert_pretrained_name_or_path = 'DeepPavlov/rubert-base-cased-conversational', vocab = vocab)

# %%

checker.finetune(clean_file = clean_file, corrupt_file = corrupt_file, 
                data_dir = data_dir, batch_size = 64, n_epochs = 4) 


# %%

test_data[5][1]

# %%

# test_data[5]
checker.correct(test_data[5][1])

# %%

for i in test_data[0:5]:
    print(i[1])
    print(checker.correct(i[1]))
    print()




# %%






# %%


for i in test_data[0:5]:
    print(i[1])
    print(checker.correct(i[1]))
    print()

# %%

checker = BertChecker(device = "cuda") 
checker.from_pretrained(
    ckpt_path = f"{check_dir}/rubert-base-cased-conversational"  # "<folder where the model is saved>"
)

# %%

print(checker.correct('в качестве прогнозирования результатов прочих проверки гипотез проверки. а нелинейных и линейных. н нелинейной и линейной информации то есть на основе. а информации делается сводные'))

# %%

checker.evaluate(clean_file = clean_file, corrupt_file = corrupt_file, data_dir = data_dir)

# %%



# %%
checker = BertChecker(device = "cuda") 
checker.from_huggingface(bert_pretrained_name_or_path = 'bert-base-cased', vocab = vocab)



# %%



# %%

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

    return correct, corrupt

# %%

a = 'Не зн аете, как разнозобр4азить вечер?' # ['Есть', 'где', '-', 'нибудь', 'отзыв', '?']
b = 'Не знаете, как разнообразить вечер?'
print(a.split())
print(b.split())

# %%

a = list(map(token_correct, a.split()))
b = list(map(token_correct, b.split()))


# %%

phrase_correct(b, a)



# %%


# %%




# %%




# %%

t1 = set('где-нибудь')
t2 = set('когда-ниб,удь')

like_tokens(t1, t2)




# %%
