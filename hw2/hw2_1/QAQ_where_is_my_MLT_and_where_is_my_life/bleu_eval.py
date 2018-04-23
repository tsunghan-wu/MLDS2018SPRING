import math
import operator
import sys
import json
from termcolor import colored
from functools import reduce 
def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t,flag = False):

    score = 0.  
    count = 0
    candidate = [s.strip()]
    if flag:
        references = [[t[i].strip()] for i in range(len(t))]
    else:
        references = [[t.strip()]] 
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score
### Usage: python bleu_eval.py caption.txt
### Ref : https://github.com/vikasnar/Bleu
if __name__ == "__main__" :
    test = json.load(open('testing_label.json','r'))
    output = sys.argv[1]
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    output = []
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        output.append((result[item['id']],score_per_video[0]))
	bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))
    import numpy as np
    output= np.array(output)
    score = output[:,1]
    asort = np.argsort(score)
    now_idx = 0
    while now_idx < len(asort):
	now_len = 0
	this_line = ""
	output[asort[now_idx]][1] = str(output[asort[now_idx]][1])[:4]
	#print output[asort[now_idx]]
	if len(str(output[asort[now_idx]])) > 95:
	 	now_idx += 1
	while now_idx < len(asort) and now_len + len(str(output[asort[now_idx]])) < 95:
		this_line = this_line + output[asort[now_idx]][0] + '('+ str(output[asort[now_idx]][1]) + ')\t'
		now_len += len(str(output[asort[now_idx]]))
		now_idx += 1
		if now_idx < len(asort):
			output[asort[now_idx]][1] = str(output[asort[now_idx]][1])[:4]
		#print output[asort[now_idx]]
	if this_line == "":
		print('now_idx:%d , len=%d'%( now_idx ,len(asort)) )
		break
	print(this_line)
        

    #for idx_idx in range(0,len(asort),2):
    #    print(str(output[asort[idx_idx]] )+"\t"+ str(output[asort[idx_idx]]))

