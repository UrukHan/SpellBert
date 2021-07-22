from tqdm import tqdm
from corus import load_ods_rt
import random

def distort(data):
  end_one = ['ами', 'ой', 'и', 'а']
  end_two = ['ая', 'ой', 'ые', 'ыми', 'ые', 'ое', 'ый']
  ind = 0
  for i in data:
    temp = ''
    for j in i.split(' '):
      if len(j) > 2:
        if j[-3:] in end_one:
          j = j[:-3] + random.choice(end_one)
        elif j[-2:] in end_one:
          j = j[:-2] + random.choice(end_one)
        elif j[-1:] in end_one:
          j = j[:-1] + random.choice(end_one)
        if j[-3:] in end_two:
          j = j[:-3] + random.choice(end_two)
        elif j[-2:] in end_two:
          j = j[:-2] + random.choice(end_two)
        elif j[-1:] in end_two:
          j = j[:-1] + random.choice(end_two)
        else:
          pass
      if len(temp) == 0:
        temp = temp + j
      else:
        temp = temp + ' '
        temp = temp + j
    for k in range(int(len(temp)/10)):
      num = random.randint(0, len(temp))
      temp = temp[:num] + random.choice(temp) + temp[num:] 
    data[ind] = temp
    ind += 1
  return data

def load_clean(path):
    records = load_ods_rt(path)

    data = []
    for record in tqdm(records):
        temp = record.text
        sim = list(""",;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{}""")

        for i in range(len(temp)):
            try:
                if temp[i] == 'n' and (len(temp[i+1].encode()) != 1 \
                    or len(temp[i-1].encode()) != 1):
                    if temp[i+1] in sim:
                        temp = temp[:i] + '' + temp[i+1:]
                    else:
                        temp = temp[:i] + ' ' + temp[i+1:]
                if temp[i] == '.' and  temp[i-1] == ' ':
                    temp = temp[:i-1] + temp[i:]

            except IndexError:
                pass

        splits = temp.split('.')

        for i in splits:
          if len(i) > 2 and i[0] == ' ':
            i = i[1:]
          if len(i) > 20 and len(i) < 300:
            ind = 0
            for k in range(len(i)):
              if k != 0 and i[k] == ' ' and i[k-1] == ' ':
                ind = 1
            if ind == 0:
              data.append(i + '.')

    return data

def clean_txt(data):
    temp_data = data

    file = open('D:/Google/NeuroNet/BERT/SpellBert/clean.txt', 'w', encoding="utf8") 
    temp = ''
    for i in tqdm(temp_data):
        if len(temp) == 0:
            temp = temp + i
        else:
            temp = temp + '\n'
            temp = temp + i

    file.write(temp) 
    file.close()
    return

def corrupt_txt(clean_path):
    opfile = open(clean_path, "r", encoding="utf8")
    corrupt_file = []
    for line in tqdm(opfile):
        if line.strip() != "": corrupt_file.append(line.strip())
    corrupt_file = distort(corrupt_file)
    file = open('D:/Google/NeuroNet/BERT/SpellBert/corrupt.txt', 'w', encoding="utf8") 
    temp = ''
    for i in tqdm(corrupt_file):
        if len(temp) == 0:
            temp = temp + i
        else:
            temp = temp + '\n'
            temp = temp + i
    file.write(temp) 
    file.close()