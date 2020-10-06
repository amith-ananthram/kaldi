#!/usr/bin/env python

import kaldi_io
import argparse
from collections import defaultdict

def write_split_xvectors(tgt_xvector_ark, tgt_xvector_scp, tgt_xvector_utt2spk, utt2xvectors, utt2spk):
	ark_scp_output='ark:| copy-vector ark:- ark,scp:%s.ark,%s.scp' % (
		tgt_xvector_ark, tgt_xvector_scp)
	with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
		for utt in utt2xvectors.keys():
			for chunk_id, xvector in enumerate(utt2xvectors[utt]):
				sub_utt = "%s-%s" % (utt, chunk_id)
				kaldi_io.write_vec_flt(f, xvector, key=sub_utt)

	with open(tgt_xvector_utt2spk, 'w') as f:
		for utt in utt2xvectors.keys():
			for chunk_id in range(len(utt2xvectors[utt])):
				sub_utt = "%s-%s" % (utt, chunk_id)
				f.write("%s %s\n" % (sub_utt, utt2spk[utt]))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Splits a series of x-vector matrices into individual vectors for LDA training.')
	parser.add_argument('--src_xvector_scp', dest='src_xvector_scp')
	parser.add_argument('--src_xvector_utt2spk', dest='src_xvector_utt2spk')
	parser.add_argument('--tgt_xvector_ark', dest='tgt_xvector_ark')
	parser.add_argument('--tgt_xvector_scp', dest='tgt_xvector_scp')
	parser.add_argument('--tgt_xvector_utt2spk', dest='tgt_xvector_utt2spk')
	parser.add_argument('--num_subsets', dest='num_subsets')
	args = parser.parse_args()

	utt2spk = {}
	with open(args.src_xvector_utt2spk, 'r') as f:
		for line in f.readlines():
			utt, spk = line.split(' ')
			utt2spk[utt] = spk

	subset = 0
	utt2xvectors = {}
	for utt, mat in kaldi_io.read_mat_scp(args.src_xvector_scp):
		utt2xvectors[utt] = mat

		if int(args.num_subsets) > 1 and len(utt2xvectors) > (len(utt2spk) / int(args.num_subsets)):
			write_split_xvectors(
				"%s.%s" % (args.tgt_xvector_ark, subset),
				"%s.%s" % (args.tgt_xvector_scp, subset),
				"%s.%s" % (args.tgt_xvector_utt2spk, subset),
				utt2xvectors,
				utt2spk
			)
			subset += 1
			utt2xvectors = {}

	if len(utt2xvectors) > 0:
		if int(args.num_subsets) > 1:
			write_split_xvectors(
				"%s.%s" % (args.tgt_xvector_ark, subset),
				"%s.%s" % (args.tgt_xvector_scp, subset),
				"%s.%s" % (args.tgt_xvector_utt2spk, subset),
				utt2xvectors,
				utt2spk
			)
		else:
			write_split_xvectors(
				args.tgt_xvector_ark,
				args.tgt_xvector_scp,
				args.tgt_xvector_utt2spk,
				utt2xvectors,
				utt2spk
			)