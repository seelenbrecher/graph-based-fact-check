# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

NAME=ka-kgat-concept-gat\#10.5.8.roberta.snopes-2

CUDA_VISIBLE_DEVICES=7 python test_roberta.py --outdir ./output/ \
 --test_path ../data/snopes_with_concepts_and_graph_roberta/test.json \
 --bert_pretrain ../checkpoint/roberta_large_mlm \
 --checkpoint ../checkpoint/$NAME/model.best.pt \
 --use_concept \
 --span_use_gat \
 --span_gat_dropout 0.0 \
 --span_gat_add_skip_conn \
 --roberta \
 --concept_dim 1024 \
 --node_dim 1024 \
 --span_gat_n_features 1024 256 256 1 \
 --batch_size 2 \
 --name asds.json

python ukp_score_test.py --predicted_labels ./output/asds.json  --predicted_evidence ../data/bert_eval.json --actual ../data/snopes_with_concepts_and_graph_roberta/test.json
