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
    parser.add_argument('out_file', metavar='result.tsv',
                        help='Write result to file.')
    parser.add_argument('begin_index', metavar='index_begin',
                        help='Begin index you want to test.', type=int)
    parser.add_argument('end_index', metavar='index_begin',
                        help='End index you want to test.', type=int)
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
    with open(OPTS.out_file, 'a') as f:
        index = 1
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(['id', 'text_a', 'text_b', 'label'])    
        end_index = OPTS.end_index if len(dataset) >= OPTS.end_index+1 else len(dataset)
        for i in range(OPTS.begin_index, end_index):
            # if i >= OPTS.amount:
            #     f.close()
            #     return
            dataline = dataset[i][0].split('\t')
            res = get_label(dataline[0], dataline[1], nlp)
            print(str(index) + '/' + str(i))
            if res == 'duplicate':
                # f.write(res + '\n')
                tsv_w.writerow([str(index), dataline[0], dataline[1], '1'])
                index += 1

if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.model == 't5':
        model_name = "PavanNeerudu/t5-base-finetuned-qqp"
        nlp = pipeline('text2text-generation', model=model_name, tokenizer=model_name, device=0)
    elif OPTS.model == 'DeBERTa':
        model_name = "Tomor0720/deberta-large-finetuned-qqp"
        nlp = pipeline('text-classification', model=model_name, tokenizer=model_name, device=0)
    else:
        NameError
    main(nlp)