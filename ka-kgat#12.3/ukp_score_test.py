import argparse
import json
import sys
from fever.scorer import fever_score
from prettytable import PrettyTable

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score


def print_metrics(gold_labels, predictions, logger=None, average='macro'):
    info = logger.info if logger is not None else print
    recall = recall_score(gold_labels, predictions, average=average)
    precision = precision_score(gold_labels, predictions, average=average)
    f1 = 2 * recall * precision / (recall + precision)
    print("Accuracy: " + str(accuracy_score(gold_labels, predictions)) + "\tRecall: " + str(
        recall) + "\tPrecision: " + str(precision) + "\tF1 " + average + ": " + str(f1))
    print("Confusion Matrix:\n" + str(confusion_matrix(gold_labels, predictions)))


def print_macro_metrics_per_confusion_matrix(cm, logger=None):
    assert len(cm) == len(cm[0]), "For a confusion matrix the lengths of both ranks should be equal"
    info = logger.info if logger is not None else print
    precisions = []
    recalls = []
    all_tp = 0
    for i in range(len(cm)):
        tp = cm[i][i]
        all_tp += tp
        fp_tp = sum(e for e in cm[i])
        fn_tp = sum(cm[j][i] for j in range(len(cm)))
        precisions.append(0 if fp_tp == 0 else tp / fp_tp)
        recalls.append(0 if fn_tp == 0 else tp / fn_tp)
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    accuracy = all_tp / sum(sum(row) for row in cm)
    print("Accuracy: " + str(accuracy) + "\tMacro Recall: " + str(macro_recall) + "\tMacro Precision: " + str(
        macro_precision) + "\tMacro F1: " + str(macro_f1))

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)

args = parser.parse_args()

predicted_labels = {}
actual = {}

ids = dict()
with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        predicted_labels[json.loads(line)["id"]] = (json.loads(line)["predicted_label"])

with open(args.actual, "r") as actual_file:
    for line in actual_file:
        actual[json.loads(line)["id"]] = (json.loads(line)["label"])

ground_truths, preds = [], []
for id in actual:
    gt = actual[id]
    pred = predicted_labels[id]
    ground_truths.append(gt)
    preds.append(pred)

prec,rec,f1,_ = precision_recall_fscore_support(ground_truths,preds,average='weighted')
print('weighted', prec, rec, f1)
print(';asd')

prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, preds, average='macro')
print('macro', prec, rec, f1)

tab = PrettyTable()
tab.field_names = ["precision", "recall", "f1_score"]
tab.add_row((round(prec,4),round(rec,4),round(f1,4)))

#conf_mat = {}
#conf_mat[0] = {'t': 0, 'fp': 0, 'fn': 0}
#conf_mat[1] = {'t': 0, 'fp': 0, 'fn': 0}
#conf_mat[2] = {'t': 0, 'fp': 0, 'fn': 0}
conf_mat = {}
conf_mat[0] = [0, 0, 0]
conf_mat[1] = [0, 0, 0]
conf_mat[2] = [0, 0, 0]

labels_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}

for pred, gt in zip(preds, ground_truths):
    pred = labels_map[pred]
    gt = labels_map[gt]
    
    #if pred == gt:
    #    conf_mat[gt]['t'] += 1
    #else:
    #    conf_mat[gt]['fn'] += 1
    #    conf_mat[pred]['fp'] += 1

    conf_mat[pred][gt] += 1

tp = 0
fn = 0
fp = 0
for i in range(0, 3):
    tp += conf_mat[i][i]
    for j in range(0, 3):
        if j == i:
            continue
        fn += conf_mat[j][i]
        fp += conf_mat[i][j]

for i in range(0, 3):
   print(conf_mat[i])
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2*prec*rec / (prec + rec)

print(tp, fp, fn)
print(prec, rec, f1)


tp = [0,0,0]
fn = [0,0,0]
fp = [0,0,0]
for i in range(0, 3):
    tp[i] = conf_mat[i][i]
    for j in range(0, 3):
        if j == i:
            continue
        fn[i] = conf_mat[j][i]
        fp[i] = conf_mat[i][j]

for i in range(0, 3):
   print(conf_mat[i])

prec = [0,0,0]
rec = [0,0,0]
f1 = [0,0,0]

for i in range(0, 3):
    prec[i] = tp[i] / (tp[i] + fp[i])
    rec[i] = tp[i] / (tp[i] + fn[i])
    f1[i] = 2 * prec[i] * rec[i] / (prec[i] + rec[i])

norm = [0,0,0]
for gt in ground_truths:
    norm[labels_map[gt]] += 1
for i in range(0, 3):
    norm[i] /= len(ground_truths)
print(norm)
print(prec)
print(rec)
print(f1)

s_prec, s_rec, s_f1 = 0,0,0
for i in range(0, 3):
    s_prec += prec[i] * norm[i]
    s_rec += rec[i] * norm[i]
    s_f1 += f1[i] * norm[i]
print(s_prec/3, s_rec/3, s_f1/3)
print('sebelomnya weighted')

print('pake dari snopes')
print_metrics(ground_truths, preds)


prec = 0
rec = 0
f1 = 0
for i in range(0, 3):
    tp = conf_mat[i]['t']
    fn = conf_mat[i]['fn']
    fp = conf_mat[i]['fp']
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    prec += p
    rec += r
    f1 += 2*p*r / (p+r)
prec /= 3
rec /= 3
f1 /= 3
    
        
print(conf_mat)

print(prec, rec, f1)
