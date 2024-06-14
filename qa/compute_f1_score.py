"""
Given two input sequences, compute the F1 score between them.
Adapted from SQuAD evaluation script. https://rajpurkar.github.io/SQuAD-explorer/
https://storageclwsprod1.blob.core.windows.net/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents.gz?se=2024-05-10T09%3A48%3A33Z&sp=rt&sv=2019-12-12&sr=b&rscd=inline%3B%20filename%3D%22evaluate-v2.0.py%22&rsce=gzip&rsct=text/x-python&sig=QqFmK6UyTyOebr26yNc4GIfmAnyM2vmTs5LysdDWpKA%3D

https://rajpurkar.github.io/SQuAD-explorer/ (distributed under the CC BY-SA 4.0 license)
"""
import collections
import re
import string

def safe_divide(numerator, denominator):
    """
    Returns 0 if denominator is zero.
    """
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
#   return white_space_fix(lower(s))
  return white_space_fix(remove_punc(lower(s)))
#   return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_f1_score(a_gold, a_pred):
    """
    My version of SQuAD F1 score computation.
    """

    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    # compute recall
    # loop over gold_toks, check if it is in pred_toks
    # if it is, increment num_same
    num_same_recall = 0
    for tok in gold_toks:
        if tok in pred_toks:
            num_same_recall += 1
    recall = safe_divide(num_same_recall, len(gold_toks))
    print("recall:", recall)

    # compute precision
    # loop over pred_toks, check if it is in gold_toks
    # if it is, increment num_same
    num_same_precision = 0
    for tok in pred_toks:
        if tok in gold_toks:
            num_same_precision += 1
    precision = safe_divide(num_same_precision, len(pred_toks))
    print("precision:", precision)

    # compute F1
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return f1, precision, recall

# def compute_f1_score(a_gold, a_pred):
#     """
#     SQuAD F1 score computation.
#     Both inputs are list of integers (tokenized sequences).
#     """
#     gold_toks = get_tokens(a_gold)
#     pred_toks = get_tokens(a_pred)

#     common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#     print("common:", common)
#     print("common.values(): {}\ttotal: {}".format(common.values(), sum(common.values())))
#     print("len(gold_toks):", len(gold_toks))
#     print("len(pred_toks):", len(pred_toks))
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

if __name__ == "__main__":
    # Test

    # Question: where is blood pumped after it leaves the right ventricle?

    gold_seq_text = "From the right ventricle , blood is pumped through the semilunar pulmonary valve into the left and right main pulmonary arteries ( one for each lung ) , which branch into smaller pulmonary arteries that spread throughout the lungs."
    pred_seq_text = "Blood pumped from the left ventricles is the only blood flow that is allowed by the left pulmonary artery. The blood from the pulmonary vein enters the left atrium, then flows through the mitral valve to the left. After the left ventilation is filled with blood the aortic valve and into the main pulmonaryartery."

    score_self, _, _ = compute_f1_score(gold_seq_text, gold_seq_text) # 1.0
    print("score (golden to golden):", score_self)

    score, _, _ = compute_f1_score(gold_seq_text, pred_seq_text) # 0.0
    print("score:", score)