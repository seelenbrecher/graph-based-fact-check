# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

# no concept. but merge some sub-word to span level, based on mentioned concept
# different with normal ka-kgat-concept-only : 
#     Linear(self.bert_hidden_dim + self.concept_dim, self.bert_hidden_dim * 2),
#     ReLU(True),
#     Linear(self.bert_hidden_dim * 2, self.node_dim)
# where normal ka-kgat concept:
#     Linear(self.bert_hidden_dim + self.concept_dim, self.node_dim),
#     ReLU(True)
CUDA_VISIBLE_DEVICES=8 python train.py --outdir ../checkpoint/ka-kgat-concept-only-3 \
--train_path ../data/fever_with_concepts/bert_train_concept.json \
--valid_path ../data/fever_with_concepts/bert_dev_concept.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--use_concept
