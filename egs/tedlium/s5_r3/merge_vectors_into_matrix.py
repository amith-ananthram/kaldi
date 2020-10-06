#!/usr/bin/env python

import kaldi_io
import argparse
import numpy as np
from collections import defaultdict

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Merges a series of x-vector vectors into matrices for TDNN training.')
	parser.add_argument('--src_xvector_scp', dest='src_xvector_scp')
	parser.add_argument('--tgt_xvector_ark', dest='tgt_xvector_ark')
	parser.add_argument('--tgt_xvector_scp', dest='tgt_xvector_scp')
	args = parser.parse_args()

	utt2xvectors = defaultdict(dict)
	for sub_utt, vec in kaldi_io.read_vec_flt_scp(args.src_xvector_scp):
		utt = ''.join(sub_utt.split('-')[:-1])
		chunk_id = int(sub_utt.split('-')[-1])
		utt2xvectors[utt][chunk_id] = vec

	ark_scp_output='ark:| copy-feats ark:- ark,scp:%s,%s' % (
		args.tgt_xvector_ark, args.tgt_xvector_scp)
	with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
		for utt in utt2xvectors.keys():
			xvector_mat = []
			for chunk_id in sorted(utt2xvectors[utt]):
				xvector_mat.append(utt2xvectors[utt][chunk_id])

			kaldi_io.write_mat(f, np.array(xvector_mat), key=utt)