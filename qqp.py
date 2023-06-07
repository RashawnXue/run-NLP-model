import argparse
import csv
import numpy as np
import sys
from transformers import pipeline

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser(
        'Script for t5(https://huggingface.co/PavanNeerudu/t5-base-finetuned-qqp) and DeBERTa(https://huggingface.co/Tomor0720/deberta-large-finetuned-qqp).\n \
        dataset: QQP.\n \
        nlp task: SSM.')
    parser.add_argument('model', metavar='model', 
                        help='Model you want to use (t5 or DeBERTa).', type=str)
    parser.add_argument('data_file', metavar='data.tsv',
                        help='Input data tsv file.')
    parser.add_argument('out_file', metavar='result.txt',
                        help='Write result to file.')
    parser.add_argument('amount', metavar='amount',
                        help='Amount of sentence pairs you want to test.', type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def t5_get_label(question1, question2, nlp):
    input_text = "qqp question1: " + question1 + "question2: " + question2
    res = nlp(input_text)
    if 'generated_text' in res[0]:
        return res[0]['generated_text']
    else:
        return ''

def deberta_get_label(question1, question2, nlp):
    input_text = question1 + " " + question2
    res = nlp(input_text)
    if res[0]['label'] == 'LABEL_0':
        return 'not_duplicate'
    elif res[0]['label'] == 'LABEL_1':
        return 'duplicate'
    else:
        return "ERROR"

def get_label(question1, question2, nlp):
    if OPTS.model == 't5':
        return t5_get_label(question1, question2, nlp)
    elif OPTS.model == 'DeBERTa':
        return deberta_get_label(question1, question2, nlp)
    else:
        return ''

def main(nlp):
    dataset = []
    with open(OPTS.data_file) as f:
        data = csv.reader(f)
        for row in data:
            if len(row) > 1 :
                row = [','.join(row)]
            dataset.append(row)
    dataset = dataset[0:]
    with open(OPTS.out_file, 'w') as f:
        for i in range(len(dataset)):
            if i >= OPTS.amount:
                f.close()
                return
            dataline = dataset[i][0].split('\t')
            res = get_label(dataline[3], dataline[4], nlp)
            f.write(res + '\n')

if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.model == 't5':
        model_name = "PavanNeerudu/t5-base-finetuned-qqp"
        nlp = pipeline('text2text-generation', model=model_name, tokenizer=model_name)
    elif OPTS.model == 'DeBERTa':
        model_name = "Tomor0720/deberta-large-finetuned-qqp"
        nlp = pipeline('text-classification', model=model_name, tokenizer=model_name)
    else:
        NameError
    main(nlp)