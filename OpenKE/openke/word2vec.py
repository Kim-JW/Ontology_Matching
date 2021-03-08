from gensim.models import Word2Vec
import multiprocessing
from gensim.models import FastText
# Coding=utf-8
# Version:python3.6.0
# Tools:Pycharm 2017.3.2
_data_ = '4/17/2020 10:11 PM'
_author_ = 'SUN'

sents = []
for line in open("./benchmarks/FB15K237/train.txt"):
    line = line.replace("\n", "")
    cons = [line.split('#')[0],line.split('#')[1],line.split('#')[2]]
    #print(line.split(',')[1])
    sents.append(cons)
#sg=0:CBOW,sg=1:skip-gram
model = Word2Vec(sents, sg=0, size=1, window=5, min_count=1,iter=100, workers=multiprocessing.cpu_count())
#model = FastText(sents, size=1, window=3, min_count=1, iter=10, min_n=3, max_n=6, word_ngrams=0)
model.save("./checkpoint/woed2vec_skip-gram.bin")

# 以写的方式打开文件，如果文件不存在，就会自动创建
file_write_obj = open('./benchmarks/FB15K237/word2vec_skip-gram_Triple2Vector.txt', 'a')
for var in sents:
    file_write_obj.writelines(str(model.wv[var[0]]))
    file_write_obj.write(',')
    file_write_obj.writelines(str(model.wv[var[1]]))
    file_write_obj.write(',')
    file_write_obj.writelines(str(model.wv[var[2]]))
    file_write_obj.write('\n')
file_write_obj.close()

print(model.wv['label'])  # obtain word vector
