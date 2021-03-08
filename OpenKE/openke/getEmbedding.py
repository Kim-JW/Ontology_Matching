from module.model import TransE, TransD, TransH
from data import TrainDataLoader, TestDataLoader
'''
# TransE Embedding
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

transe = TransE(ent_tot=train_dataloader.get_ent_tot(), rel_tot=train_dataloader.get_rel_tot(), dim=1, p_norm=1,
                norm_flag=False)
matrix = transe.load_checkpoint('./checkpoint/transe_100.ckpt')

print(transe.get_parameters("numpy").keys())

ent_vals = list(transe.get_parameters().get("ent_embeddings.weight"))
ent_keys = []
for line in open("./benchmarks/FB15K237/entity2id.txt"):
    if line != "14541\n":
        ent_keys.append(line.split("\t")[0])

rel_vals = transe.get_parameters().get("rel_embeddings.weight")
rel_keys = []
for line in open("./benchmarks/FB15K237/relation2id.txt"):
    if line != "237\n":
        rel_keys.append(line.split("\t")[0])

rel_dic = dict(zip(rel_keys, rel_vals))
ent_dic = dict(zip(ent_keys, ent_vals))

f = open('./benchmarks/FB15K237/transE_Triple2Vector_100.txt', 'a')
for line in open("./benchmarks/FB15K237/train.txt"):
    line = line.replace("\n", "")
    # rel_keys.append(line.split("\t")[0])
    # print(ent_dic.get(line.split("#")[0]))
    # print(rel_dic.get(line.split("#")[1]))
    # print(ent_dic.get(line.split("#")[2]))
    f.write(str(ent_dic.get(line.split("#")[0])) + "," + str(rel_dic.get(line.split("#")[1])) + "," + str(
        ent_dic.get(line.split("#")[2])) + "\n")

f.close()
'''
'''
# TransD Embedding
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

transd = TransD(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim_e=1,
    dim_r=1,
    p_norm=1,
    norm_flag=True)

matrix = transd.load_checkpoint('./checkpoint/transd_100.ckpt')

print(transd.get_parameters("numpy").keys())

ent_vals = list(transd.get_parameters().get("ent_embeddings.weight"))
ent_keys = []
for line in open("./benchmarks/FB15K237/entity2id.txt"):
    if line != "14541\n":
        ent_keys.append(line.split("\t")[0])

rel_vals = transd.get_parameters().get("rel_embeddings.weight")
rel_keys = []
for line in open("./benchmarks/FB15K237/relation2id.txt"):
    if line != "237\n":
        rel_keys.append(line.split("\t")[0])

rel_dic = dict(zip(rel_keys, rel_vals))
ent_dic = dict(zip(ent_keys, ent_vals))

f = open('./benchmarks/FB15K237/transD_Triple2Vector_100.txt', 'a')
for line in open("./benchmarks/FB15K237/train.txt"):
    line = line.replace("\n", "")
    # rel_keys.append(line.split("\t")[0])
    # print(ent_dic.get(line.split("#")[0]))
    # print(rel_dic.get(line.split("#")[1]))
    # print(ent_dic.get(line.split("#")[2]))
    f.write(str(ent_dic.get(line.split("#")[0])) + "," + str(rel_dic.get(line.split("#")[1])) + "," + str(
        ent_dic.get(line.split("#")[2])) + "\n")

f.close()
'''

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)
# define the model

transh = TransH(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=1,
    p_norm=1,
    norm_flag=True)
# change 1 transh_100
matrix = transh.load_checkpoint('./checkpoint/transh_100.ckpt')

print(transh.get_parameters("numpy").keys())

ent_vals = list(transh.get_parameters().get("ent_embeddings.weight"))
ent_keys = []
for line in open("./benchmarks/FB15K237/entity2id.txt"):
    if line != "14541\n":
        ent_keys.append(line.split("\t")[0])

rel_vals = transh.get_parameters().get("rel_embeddings.weight")
rel_keys = []
for line in open("./benchmarks/FB15K237/relation2id.txt"):
    if line != "237\n":
        rel_keys.append(line.split("\t")[0])

rel_dic = dict(zip(rel_keys, rel_vals))
ent_dic = dict(zip(ent_keys, ent_vals))

# change2 transH_Triple2Vector_100
f = open('./benchmarks/FB15K237/transH_Triple2Vector_100.txt', 'a')
for line in open("./benchmarks/FB15K237/train.txt"):
    line = line.replace("\n", "")
    # rel_keys.append(line.split("\t")[0])
    # print(ent_dic.get(line.split("#")[0]))
    # print(rel_dic.get(line.split("#")[1]))
    # print(ent_dic.get(line.split("#")[2]))
    f.write(str(ent_dic.get(line.split("#")[0])) + "," + str(rel_dic.get(line.split("#")[1])) + "," + str(
        ent_dic.get(line.split("#")[2])) + "\n")

f.close()

