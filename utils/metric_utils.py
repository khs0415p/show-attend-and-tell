from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def get_bleu_score(refs, hyps, tokenizer, n):
    refs = [[tokenizer.tokenize(tokenizer.decode(refs[i].tolist()))] for i in range(refs.size(0))]
    hyps = [tokenizer.tokenize(tokenizer.decode(hyps[i].tolist())) for i in range(hyps.size(0))]
    score = corpus_bleu(refs, hyps, weights=(1/n,)*n, smoothing_function=SmoothingFunction().method1)
    return score