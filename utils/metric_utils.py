from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def get_metrics(refs, hyps, tokenizer):
    refs = [[tokenizer.tokenize(tokenizer.decode(refs[i].tolist()))] for i in range(refs.size(0))]
    hyps = [tokenizer.tokenize(tokenizer.decode(hyps[i].tolist())) for i in range(hyps.size(0))]

    bleu2_score = get_bleu_score(refs, hyps, 2)
    bleu4_score = get_bleu_score(refs, hyps, 4)
    meteor_score = get_meteor_score(refs, hyps)

    return bleu2_score, bleu4_score, meteor_score


def get_bleu_score(refs, hyps, n):
    score = corpus_bleu(refs, hyps, weights=(1/n,)*n, smoothing_function=SmoothingFunction().method1)
    return score


def get_meteor_score(refs, hyps):
    return meteor_score(refs, hyps)