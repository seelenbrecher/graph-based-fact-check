# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

NAME=ka-kgat-concept-gat\#2.4

rm ../checkpoint/$NAME/train_log.txt

# use concept + gat
# actually 1 layers, since the last layer = pred layer, and we skip that one
CUDA_VISIBLE_DEVICES=14 python train.py --outdir ../checkpoint/$NAME \
--train_path ../data/fever_with_concepts/bert_train_concept.json \
--valid_path ../data/fever_with_concepts/bert_eval_concept.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--use_concept \
--span_use_gat \
--num_train_epochs 5 \
--span_gat_dropout 0.0 \
--span_gat_n_layers 2 \
--span_gat_n_heads 4 1 \
--span_gat_n_features 100 192 1

CUDA_VISIBLE_DEVICES=14 python test.py --outdir ./output/ \
--test_path ../data/fever_with_concepts/bert_eval_concept.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/$NAME/model.best.pt \
--use_concept \
--span_use_gat \
--span_gat_dropout 0.0 \
--span_gat_n_layers 2 \
--span_gat_n_heads 4 1 \
--span_gat_n_features 100 192 1 \
--name $NAME-dev.json

python fever_score_test.py --predicted_labels ./output/$NAME-dev.json  --predicted_evidence ../data/bert_eval.json --actual ../data/dev_eval.json
