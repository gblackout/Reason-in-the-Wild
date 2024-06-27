# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import argparse
from metrics import bleu
from metrics import weighted_ngram_match
from metrics import syntax_match
from metrics import dataflow_match
from typing import List
from dataclasses import dataclass


py_kws = """False
None
True
and
as
assert
async
await
break
class
continue
def
del
elif
else
except
finally
for
from
global
if
import
in
is
lambda
nonlocal
not
or
pass
raise
return
try
while
with
yield"""


@dataclass
class CodeBLEUOutput:
    ngram_match_score: float
    weighted_ngram_match_score: float
    syntax_match_score: float
    dataflow_match_score: float

    alpha: float
    beta: float
    gamma: float
    theta: float

    score: float


def compute_one(pred: str, gt: str, weights:List):
    alpha, beta, gamma, theta = weights

    hypothesis = [pred]
    references = [[gt]]

    tokenized_hyps = [pred.split()]
    tokenized_refs = [[gt.split()]]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    keywords = py_kws.split('\n')

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [[reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference]
        for reference in tokenized_refs
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, 'python')

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, 'python')

    # in any case if dataflow_match_score is None, we reweight the rest and compute with them
    if dataflow_match_score is None:
        s = alpha + beta + gamma
        ra, rb, rg = alpha / s, beta / s, gamma / s
        score = (
                ra * ngram_match_score +
                rb * weighted_ngram_match_score +
                rg * syntax_match_score
        )

    else:
        score = (
                alpha * ngram_match_score +
                beta * weighted_ngram_match_score +
                gamma * syntax_match_score +
                theta * dataflow_match_score
        )

    return CodeBLEUOutput(
        ngram_match_score,
        weighted_ngram_match_score,
        syntax_match_score,
        dataflow_match_score,
        alpha,
        beta,
        gamma,
        theta,
        score=score
    )




