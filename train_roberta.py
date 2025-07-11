import argparse
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import inference_model
from data_loader import DataLoader
from prepare_concept import add_concept_args, load_transe_emb, add_span_gat_args
from sklearn.metrics import precision_recall_fscore_support
from transformers import *
from tqdm import tqdm

logger = logging.getLogger(__name__)

def reproducible():
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model_with_f1(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    ground_truths = []
    preds = []
    for index, data in enumerate(validset_reader):
        inputs, lab_tensor = data
        class_label, _, _ = lab_tensor
        probs, _, _ = model(inputs, roberta=True)
        pred = probs.max(1)[1].type_as(class_label)
        ground_truths.extend(class_label)
        preds.extend(pred)

    ground_truths = [i.item() for i in ground_truths]
    preds = [i.item() for i in preds]
    prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, preds, average='macro')
    logger.info('prec = {}, rec={}, f1={}'.format(prec, rec, f1))
    return f1

def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for index, data in enumerate(validset_reader):
        inputs, lab_tensor = data
        class_label, _, _ = lab_tensor
        probs, _, _ = model(inputs, roberta=True)
        correct_pred += correct_prediction(probs, class_label)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy



def train_model(model, ori_model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_accuracy = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    no_grad = ["embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2",
               "encoder.layer.3","encoder.layer.4","encoder.layer.5","encoder.layer.6",
                "encoder.layer.7", "encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11"]
    no_grad = []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_grad):
            p.requires_grad = False
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        optimizer.zero_grad()
        for index, data in enumerate(trainset_reader):
            inputs, lab_tensor = data
            class_label, evi_labels, _ = lab_tensor
            probs, evi_probs, _ = model(inputs, roberta=True)
            loss = F.nll_loss(probs, class_label)
            if args.use_evi_select_loss:
                loss += F.binary_cross_entropy(evi_probs, evi_labels)
            running_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                with torch.no_grad():
                    if args.with_f1:
                        dev_accuracy = eval_model_with_f1(model, validset_reader)
                    else:
                        dev_accuracy = eval_model(model, validset_reader)
                    logger.info('Dev total acc: {0}'.format(dev_accuracy))
                    if dev_accuracy > best_accuracy:
                        best_accuracy = dev_accuracy

                        torch.save({'epoch': epoch,
                                    'model': ori_model.state_dict(),
                                    'best_accuracy': best_accuracy}, save_path + ".best.pt")
                        logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_accuracy))



if __name__ == "__main__":
    reproducible()
    parser = argparse.ArgumentParser()
    parser = add_concept_args(parser)
    parser = add_span_gat_args(parser)
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=1024, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--kernel", type=int, default=21, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=130, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--postpretrain')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--roberta', action='store_true', default=False)
    parser.add_argument('--use_evi_select_loss', action='store_true', default=False)
    parser.add_argument('--with_f1', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, label_map, tokenizer, args,
                                 batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, label_map, tokenizer, args,
                                 batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')
    bert_model = AutoModel.from_pretrained(args.bert_pretrain)
    if args.bert_pretrain:
        model_dict = bert_model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.bert_pretrain, "pytorch_model.bin"))
        pretrained_dict = {k.replace("roberta.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print (pretrained_dict.keys())
        model_dict.update(pretrained_dict)

    logger.info('loading transe model')
    concept_model = None
    
    ori_model = inference_model(bert_model, concept_model, args)
    model = nn.DataParallel(ori_model)
    model = model.cuda()
    train_model(model, ori_model, args, trainset_reader, validset_reader)
