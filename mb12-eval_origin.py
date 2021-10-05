#!/usr/bin/env python2.7

import argparse
import collections
import gzip
import numpy as np
import pandas as pd
import random
import EvalJig as ej
import os

# apar = argparse.ArgumentParser(description='Evaluation script for TREC 2012 Microblog Track runs')
# apar.add_argument('qrels', help='Relevance judgments file')
# apar.add_argument('run', help='System output file')
# apar.add_argument('-v', '--verbose', action='store_false', help='Be verbose')
# apar.add_argument('-M', dest='eval_depth', type=int, default=1000,
#                   help='Maximum evaluation depth (default: 1000)')
# apar.add_argument('-l', dest='min_rel', type=int, default=2,
#                   help='Minimum rel value to count as relevant (default: 2)')
# args = apar.parse_args()



class ROC:
    """ Compute a "truncated" ROC curve.  Since the rankings only go down to
    rank 10,000, and in trec_eval we have a common measurement depth of 1000,
    the run does not in fact score the entire collection.  Rather than play
    statistical games with the remainder of the ranking, we just generate an ROC
    curve which maxes out at a FP rate around 1/1000 the size of the collection."""

    def __init__(self, *args, **kwargs):
        self.tpr = ()
        self.fpr = ()
        self.score = ()
        if (kwargs):
            self.compute(kwargs['ranking'], kwargs['qrel'], kwargs['minrel'])

    def compute(self, ranking, qrel, minrel):
        ndocs = 16141812
        self.score = [score for score, doc in ranking]
        num_rel = sum(1 for rel in qrel.values() if rel >= minrel)
        num_nonrel = ndocs - num_rel
        if num_rel == 0:
            self.tpr = [0.0 for score in ranking]
            self.fpr = [1.0 for score in ranking]
        elif ndocs - num_rel == 0:
            self.tpr = [1.0 for score in ranking]
            self.fpr = [0.0 for score in ranking]
        else:
            tp = 0
            fp = 0
            self.tpr = np.zeros(len(ranking))
            self.fpr = np.zeros(len(ranking))
            i = 0
            for (score, docid) in ranking:
                if (docid not in qrel or qrel[docid] < minrel):
                    fp += 1
                else:
                    tp += 1
                self.fpr[i] = float(fp) / num_nonrel
                self.tpr[i] = float(tp) / num_rel
                i += 1

    def as_str(self, lead):
        return "\n".join(
            (" ".join((lead, '[', str(fp), str(tp), str(sc), ']')) for tp, fp, sc in
             zip(self.tpr, self.fpr, self.score)))


class ROC_Plot(ej.Measure):
    def __str__(self):
        return "roc"

    def compute(self, ranking, qrel, minrel):
        roc = ROC(ranking=ranking, qrel=qrel, minrel=minrel)
        return roc

    def pretty(self, topic, roc):
        return roc.as_str(str(topic) + " roc")

    def pretty_mean(self, roc):
        return roc.as_str("all roc")

    def redux(self, scores):
        # find the overall minimum and maximum scores
        minsc = 1000
        maxsc = -1000
        for roc in scores.values():
            minsc = min(minsc, min(roc.score))
            maxsc = max(maxsc, max(roc.score))

        # Threshold average:
        # (analogous to the RP curve interpolation.)
        # At a common series of score thresholds, average the (tp,fp)
        # at the score equal to or less than the threshold.
        numpts = len(scores.values()[0].score)
        fprsum = np.zeros(numpts)
        tprsum = np.zeros(numpts)
        score = np.zeros(numpts)
        interval = (maxsc - minsc) / numpts
        for roc in scores.values():
            i = 0
            thresh = maxsc
            for j in range(0, numpts):
                while i < (len(roc.score) - 1) and roc.score[i] > thresh:
                    i += 1
                fprsum[j] += roc.fpr[i]
                tprsum[j] += roc.tpr[i]
                score[j] = thresh
                thresh -= interval
        for j in range(0, numpts):
            fprsum[j] /= len(scores)
            tprsum[j] /= len(scores)
        roc = ROC()
        (roc.tpr, roc.fpr, roc.score) = tprsum, fprsum, score
        return roc


class AUC(ej.Measure):
    """This is an area-under-the-ROC-curve measure computing using the Mann-
    Whitney-Wilcoxon statistic.  Because we have these unusual scorings of the
    collection with millions of tweets tied at rank 10,000, standard interpolative
    methods for computing AUC behave strangely.
    I'm not completely happy with this one yet, either."""

    def __str__(self):
        return "mw_auc"

    def __init__(self):
        self.formatstr = '{:f}'

    def compute(self, ranking, qrel, minrel):
        rank = 0
        R = 0
        rel_ret = 0
        for (score, id) in ranking:
            rank += 1
            if id in qrel and qrel[id] >= minrel:
                R += rank
                rel_ret += 1
        if rel_ret == 0: return 0.0
        num_rel = sum(1 for rel in qrel.values() if rel >= minrel)
        if (rel_ret < num_rel):
            # All nonretrieved relevant are retrieved tied at the end of the ranking
            R += len(ranking) + 1
        ndocs = len(ranking) + 1
        num_nonrel = ndocs - num_rel
        U = R - (num_rel * (num_rel + 1)) / 2.0
        return U / (num_rel * num_nonrel)


## jig.add_op(AUC())
# jig.add_op(ROC_Plot())

limit_to_rec_topics = False
path = 'active_retrieval_experiment/' #path to dir containing result files
qrels_file = 'adhoc-qrels_filtered' #path to labels

for file in os.listdir(path):
    if '.txt' in file:
        run_name = os.path.join(path, file)
        print(run_name)
        # df = pd.read_csv(run_name, names=['topic', 'docid', 'score', 'tag'], header=None, sep=' ')
        # df.sort_values('score', ascending=True).to_csv(run_name, sep=' ', header=None, index=False)
        iteration = file[file.index('iter'):].split('_')[1]
        Args = collections.namedtuple('Args', ['qrels', 'run', 'verbose', 'eval_depth', 'min_rel'])
        args = Args(qrels_file, run_name, 0, 1000, 1)
        jig = ej.EvalJig()
        jig.add_op(ej.NumRetr())
        jig.add_op(ej.NumRel())
        jig.add_op(ej.RelRet())
        jig.add_op(ej.R_Precision())
        jig.add_op(ej.AvePrec())
        jig.add_op(ej.PrecAt(10))
        jig.add_op(ej.PrecAt(20))
        jig.add_op(ej.PrecAt(30))
        jig.add_op(ej.MRE())
        jig.add_op(ej.RelString())
        jig.minrel = args.min_rel
        jig.evaldepth = args.eval_depth
        jig.verbose = args.verbose

        qrels = collections.defaultdict(dict)
        with open(args.qrels) as qrelsfile:
            for line in qrelsfile:
                topic, _, docid, rel = line.split()
                qrels[topic][docid] = int(rel)

        run = collections.defaultdict(list)
        with open(args.run) as runfile:
            for line in runfile:
                if len(line.split()) == 4:
                    topic, docid, score, tag = line.split()
                else:
                    topic, docid, tag = line.split()
                    score = '1000.0'
                topic = topic.lstrip('MB0')
                if not limit_to_rec_topics:
                    run[topic].append((float(score), docid))
                elif int(topic) in topics_in_rec:
                    run[topic].append((float(score), docid))

        for topic in run.keys():
            if topic in qrels:
                jig.compute(topic, run[topic], qrels[topic])

        jig.print_scores()
        jig.comp_means()
        jig.print_means()

        row = jig.get_means()
        row['method_name'] = run_name
        row['iter'] = iteration

        rows = [row]
        # output_dir = 'output/'
        output_dir = path
        output_file_name = os.path.join(output_dir, 'results_v1.xlsx')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(output_file_name):
            df = pd.read_excel(output_file_name)
        else:
            df = pd.DataFrame()
        new_df = pd.DataFrame(rows)
        df.append(new_df).to_excel(output_file_name, index=False)
