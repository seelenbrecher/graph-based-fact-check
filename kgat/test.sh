# python test.py --outdir ./output \
# --test_path ../data/bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/kgat/model.best.pt \
# --name dev.json

# python test.py --outdir ./output \
# --test_path ../data/bert_test.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/kgat/model.best.pt \
# --name test.json



CUDA_VISIBLE_DEVICES=9 python test.py --outdir ./output/ \
--test_path ../data/bert_eval.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/reproduce-kgat-1/model.best.pt \
--name asd.json

#CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
#--test_path ../data/bert_test.json \
#--bert_pretrain ../bert_base \
#--checkpoint ../checkpoint/reproduce-kgat/model.best.pt \
#--name rep-test.json
