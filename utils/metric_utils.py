from itertools import chain
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score


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
    refs = list(chain(*refs))
    assert len(refs) == len(hyps)

    total_score = 0
    for ref, hyp in zip(refs, hyps):
        total_score += single_meteor_score(ref, hyp)
    return total_score / len(refs)