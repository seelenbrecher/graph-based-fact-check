# do not run
# create multiple tmux, and run it 1b1, because it takes too long to run

# BERT train
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_train_concept.json --output ../fever_with_concepts_and_graph.5.7/train.json
 
 # BERT dev
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_dev_concept.json --output ../fever_with_concepts_and_graph.5.7/dev.json

# BERT eval
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_eval_concept.json --output ../fever_with_concepts_and_graph.5.7/eval.json

# BERT test
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_test_concept.json --output ../fever_with_concepts_and_graph.5.7/test.json

# ROBERTA train
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_train_concept.json --output ../fever_with_concepts_and_graph_roberta.5.7/train  --bert_pretrain ../../checkpoint/roberta_large_mlm  --roberta

# ROBERTA dev
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_dev_concept.json --output ../fever_with_concepts_and_graph_roberta.5.7/dev --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta


# ROBERTA eval
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_eval_concept.json --output ../fever_with_concepts_and_graph_roberta.5.7/eval --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta

# ROBERTA test
python construct_concept_graph.v1.py --input ../fever_with_concepts/bert_test_concept.json --output ../fever_with_concepts_and_graph_roberta.5.7/test --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta


# snopes-BERT train
python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/train.json --output ../snopes_with_concepts_and_graph_clean.5.7/train
 
 # snopes-BERT dev
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/dev.json --output ../snopes_with_concepts_and_graph_clean.5.7/dev
 
 # snopes-BERT eval
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/test.json --output ../snopes_with_concepts_and_graph_clean.5.7/test
 
 # snopes-ROBERTA train
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/train.json --output ../snopes_with_concepts_and_graph_roberta_clean.5.7/train --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta
 
 # snopes-roberta-dev
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/dev.json --output ../snopes_with_concepts_and_graph_roberta_clean.5.7/dev --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta
 
 # snopes-roberta-test
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk5/test.json --output ../snopes_with_concepts_and_graph_roberta_clean.5.7/test --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta

############################### snopes filter
# snopes-BERT train
python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/train.json --output ../snopes_with_concepts_and_graph_filter.5.7/train
 
 # snopes-BERT dev
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/dev.json --output ../snopes_with_concepts_and_graph_filter.5.7/dev
 
 # snopes-BERT eval
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/test.json --output ../snopes_with_concepts_and_graph_filter.5.7/test
 
 # snopes-ROBERTA train
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/train.json --output ../snopes_with_concepts_and_graph_roberta_filter.5.7/train --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta
 
 # snopes-roberta-dev
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/dev.json --output ../snopes_with_concepts_and_graph_roberta_filter.5.7/dev --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta
 
 # snopes-roberta-test
 python construct_concept_graph.v1.py --input ../snopes_with_concepts_and_graph_roberta.bk6/test.json --output ../snopes_with_concepts_and_graph_roberta_filter.5.7/test --bert_pretrain ../../checkpoint/roberta_large_mlm --roberta
