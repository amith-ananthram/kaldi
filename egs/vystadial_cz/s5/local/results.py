#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
from __future__ import division
from __future__ import print_function
import argparse
import glob
import sys
import sqlite3


def extract_stat(wer_file):
    wer, ser = None, None
    try:
        with open(wer_file, 'r') as f:
            s = f.readlines()
            wer = float(s[1].split()[1])
            ser = float(s[2].split()[1])

    except Exception as e:
        print(sys.stderr, 'Error parsing file %s' % wer_file)
        print(sys.stderr, str(e))
    return wer, ser


def extractResults(path):
    wer_files = glob.glob('%s/*/decode_*/*wer_*' % path)
    table = []
    for wf in wer_files:
        try:
            exp, decode_dir, wer_f = wf.split('/')[-3:]
            # last split: decode_it3_dev_build0  -> (dev, build0)
            lm = decode_dir.split('_')[-1]
            dataset = decode_dir.split('_')[-2]
            lm_w = int(wer_f[4:])  # strip wer_ from wer_19
            wer, ser = extract_stat(wf)
            table.append((exp, dataset, lm,  lm_w, wer, ser))
        except Exception as e:
            print('failed to parse %s' % wf, file=sys.stderr)
            print(str(e), file=sys.stderr)
    return table


class Table(object):

    def __init__(self, data=[], colnames=[]):
        self.data = data
        self.colnames = colnames
        self.colSep = '\t'
        self.lineSep = '\n'

    def data2str(self):
        strdata = []
        for r in self.data:
            strdata.append([str(c) for c in r])
        return strdata

    def __str__(self):
        sd = self.data2str()
        colwidth = [len(c) for c in self.colnames]
        for j in range(len(colwidth)):
            for r in sd:
                colwidth[j] = max(colwidth[j], len(r[j]))

        gaps = [m - len(c) for (m, c) in zip(colwidth, self.colnames)]
        rows = [self.colSep.join(
            [c + ' ' * gap for c, gap in zip(self.colnames, gaps)])]
        for r in sd:
            gaps = [m - len(c) for (m, c) in zip(colwidth, r)]
            rows.append(
                self.colSep.join([c + ' ' * d for c, d in zip(r, gaps)]))
        return self.lineSep.join(rows)


class LatexTable(Table):

    def __init__(self, data=[], colnames=[]):
        Table.__init__(self, data, colnames)
        nc = len(colnames)
        self.header = '\\begin{tabular}{%s}' % ('c' * nc)
        self.tail = '\\end{tabular}'
        self.colSep = ' & '
        self.lineSep = '\\\\ \n'

    def __str__(self):
        table_s = super(LatexTable, self).__str__()
        table_s = table_s.replace('_', '\_')
        return '%s\n%s\n%s\n' % (self.header, table_s, self.tail)


def Table2LatexTable(table):
    return LatexTable(table.data, table.colnames)


def createSmallTable(r):
    d = []
    for k, v in r.items():
        w, s, r = v
        if w == []:
            minw = None
        else:
            minw = min(w)  # returns tuple if s is list of tuples
        if s == []:
            mins = None
        else:
            mins = min(s)  # returns tuple if s is list of tuples
        mean_r = float(sum(r)) / len(r)
        d.append([k, mean_r, minw, mins])
    t = Table(d, ['exp', 'RT coef', 'WER', 'SER'])
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse experiment directory generated by kaldi vystadial recipe and print statistics')

    parser.add_argument('expath', type=str, action='store',
                        help='Path to experiment directory')
    parser.add_argument('-l', '--latex', default=False, action='store_true',
                        help='Generate also latex format table')
    args = parser.parse_args()

    raw_d = extractResults(args.expath)

    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE results (exp text, dataset text, lm text, lm_w int, wer float, ser float)''')
    c.executemany('INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)', raw_d)

    # get all results sorted
    # c.execute("SELECT * FROM results ORDER BY exp, dataset, lm, lm_w")
    # d = c.fetchall()
    # t = Table(data=d, colnames=['exp', 'set', 'lm', 'LMW', 'WER', 'SER'])
    # print '%s\n==================' % str(t)

    # best experiment
    # c.execute("SELECT exp, dataset, lm_w,  MIN(wer), ser FROM results ORDER BY exp, lm_w, dataset")
    # d = c.fetchall()
    # compare dev and test set by picking up the best experiment
    # c.execute(("SELECT exp, dataset, lm_w,  MIN(wer), ser FROM results "
    #            "GROUP BY exp, lm, dataset ORDER BY exp, lm, dataset"))
    # d = c.fetchall()
    # t = Table(data=d, colnames=['exp', 'set', 'lm', 'LMW', 'WER', 'SER'])
    # print '%s\n==================' % str(t)

    # traditional usage of devset
    dev_set_query = ("SELECT r.exp, r.lm, r.lm_w FROM results AS r "
                     "INNER JOIN ( SELECT dataset, exp, lm, MIN(wer) as min_wer "
                     "           FROM results WHERE dataset=? GROUP BY exp, lm) i "
                     "ON r.exp=i.exp AND r.lm=i.lm AND r.dataset=i.dataset AND r.wer <= i.min_wer "
                     )
    c.execute(dev_set_query, ('dev',))

    min_dev = c.fetchall()

    # remove duplicates: duplicates if equal mimimum wer in dev set
    min_dev_un = [(e, lm, lmw) for ((e, lm), lmw) in
                  list(dict([((e, lm), lmw) for e, lm, lmw in min_dev]).items())]
    # sort according LM -> sort results according experiment & LMs
    min_dev_un.sort(key=lambda x: (x[1], x[0]))

    # extract corresponding test results to dev set
    d = []
    for exp, lm, lm_w in min_dev_un:
        c.execute(("SELECT * FROM results WHERE "
                   "dataset='test' AND exp=? AND lm=? AND lm_w=?"),
                  (exp, lm, lm_w))
        x = c.fetchall()
        assert (len(x) == 1), "One row should be extracted."
        d.append(x[0])

    t = Table(data=d, colnames=['exp', 'set', 'LM', 'LMW', 'WER', 'SER'])
    print(str(t))
    if args.latex:
        print(Table2LatexTable(t))
