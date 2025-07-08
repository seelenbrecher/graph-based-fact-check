# fever_with_concepts_and_graph_1 = without pooling. lets say A and B has conceptual relation, where A = [a1, a2], B = [b1, b2], relation = r
# we add 4 edges, a1,r,b1, a1,r,b2, a2,r,b1, a2,r,b2

python modify_concept_graph.py --concept_input ../fever_with_concepts_and_graph/train.json \
--output ../fever_with_concepts_and_graph_1/train.json

python modify_concept_graph.py --concept_input ../fever_with_concepts_and_graph/eval.json \
--output ../fever_with_concepts_and_graph_1/eval.json

python modify_concept_graph.py --concept_input ../fever_with_concepts_and_graph/test.json \
--output ../fever_with_concepts_and_graph_1/test.json