# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

NAME=ka-kgat-concept-gat\#12.5.roberta

rm ../checkpoint/$NAME/train_log.txt

 # use concept + gat
CUDA_VISIBLE_DEVICES=10 python train_roberta.py --outdir ../checkpoint/$NAME \
--train_path ../data/fever_with_concepts_and_graph_roberta_with_sents_labels.5.7/extended_train_1.json \
--valid_path ../data/fever_with_concepts_and_graph_roberta_with_sents_labels.5.7/dev.json \
--bert_pretrain ../checkpoint/roberta_large_mlm \
--postpretrain ../checkpoint/roberta_large_mlm \
--use_concept \
--span_use_gat \
--num_train_epochs 2 \
--span_gat_add_skip_conn \
--span_gat_dropout 0.0 \
--concept_dim 1024 \
--node_dim 1024 \
--roberta \
--use_evi_select_loss \
--span_gat_n_features 1024 256 256 1

CUDA_VISIBLE_DEVICES=10 python test_roberta.py --outdir ./output/ \
 --test_path ../data/fever_with_concepts_and_graph_roberta_with_sents_labels.5.7/eval.json \
 --bert_pretrain ../checkpoint/roberta_large_mlm \
 --checkpoint ../checkpoint/$NAME/model.best.pt \
 --use_concept \
 --span_use_gat \
 --span_gat_dropout 0.0 \
 --span_gat_add_skip_conn \
 --roberta \
 --concept_dim 1024 \
 --node_dim 1024 \
 --batch_size 2 \
 --span_gat_n_features 1024 256 256 1 \
 --name $NAME-dev.json

python fever_score_test.py --predicted_labels ./output/$NAME-dev.json  --predicted_evidence ../data/bert_eval.json --actual ../data/dev_eval.json

CUDA_VISIBLE_DEVICES=10 python test_roberta.py --outdir ./output/ \
--test_path ../data/fever_with_concepts_and_graph_roberta_with_sents_labels.5.7/test.json \
--bert_pretrain ../checkpoint/roberta_large_mlm \
--checkpoint ../checkpoint/$NAME/model.best.pt \
--use_concept \
--span_use_gat \
--span_gat_dropout 0.0 \
--span_gat_add_skip_conn \
--roberta \
--concept_dim 1024 \
--node_dim 1024 \
--batch_size 2 \
--span_gat_n_features 1024 256 256 1 \
--name $NAME-test.json
