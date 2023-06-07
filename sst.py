import argparse
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

OPTS = None
model = None
tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for t5 and distilbert.\n \
        dataset: sst2 \n \
        nlp task: sentiment analysis.')
    parser.add_argument('model', metavar='model',
                        help='Model you want to use (t5 or DeBERTa or distilbert).', type=str)
    parser.add_argument('data_file', metavar='data.txt',
                        help='Input data txt file.')
    parser.add_argument('out_file', metavar='result.txt',
                        help='Write result to file.')
    parser.add_argument('amount', metavar='amount',
                        help='Amount of sentences you want to test.', type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def t5_get_sentiment(text):
    labels = ["p", "n"]
    class_ids = torch.LongTensor(tokenizer(labels, padding=True).input_ids)
    text = 'I hate you'
    inputs = tokenizer("sentiment: " + text, max_length=128, truncation=True, return_tensors="pt")
    preds = model.generate(inputs.input_ids.to('cpu'),
                           attention_mask=inputs.attention_mask.to('cpu'),
                           output_scores=True,
                           return_dict_in_generate=True,
                           min_length=class_ids.shape[1] + 1,
                           max_length=class_ids.shape[1] + 1,
                           do_sample=False)
    scores = torch.stack(preds.scores, dim=1).to("cpu")
    score_of_labels = scores.gather(dim=2, index=class_ids.T.expand(1, -1, -1))
    probabilities = score_of_labels.nanmean(dim=2).softmax(1)
    decoded_preds = tokenizer.batch_decode(sequences=preds[0], skip_special_tokens=True)
    print(f'negative probabilities:{probabilities[0][0].item()}, positive probabilities:{probabilities[0][1].item()}')
    if decoded_preds[0] == 'p':
        return 'POSITIVE'
    else:
        return 'NEGATIVE'


def distilbert_get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


def deberta_get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    if model.config.id2label[predicted_class_id] == 'positive':
        return 'POSITIVE'
    elif model.config.id2label[predicted_class_id] == 'negative':
        return 'NEGATIVE'
    else:
        return ''


def get_sentiment(text):
    if OPTS.model == 't5':
        return t5_get_sentiment(text)
    elif OPTS.model == 'distilbert':
        return distilbert_get_sentiment(text)
    elif OPTS.model == 'DeBERTa':
        return deberta_get_sentiment(text)
    else:
        return ''


def main():
    dataset = []
    with open(OPTS.data_file) as file:
        for line in file:
            columns = line.split('\t')
            if len(columns) == 2:
                dataset.append(columns[1])
        file.close()
    i = 0
    with open(OPTS.out_file, 'w') as f:
        for sentence in dataset:
            sentiment = get_sentiment(sentence)
            i += 1
            print(f'complete: {i} sentiment: {sentiment}')
            f.write(sentiment + '\n')
            if (i >= OPTS.amount):
                return


if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.model == 't5':
        model_name = "michelecafagna26/t5-base-finetuned-sst2-sentiment"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif OPTS.model == 'distilbert':
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    elif OPTS.model == 'DeBERTa':
        model_name = 'mrm8488/deberta-v3-small-finetuned-sst2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        NameError
    main()
