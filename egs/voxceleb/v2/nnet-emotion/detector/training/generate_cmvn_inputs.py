#!/usr/bin/env python

import os
import argparse
from collections import defaultdict

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Generate Kaldi input files for CMVN given a source utt2spk file.")
	parser.add_argument('--source_utt2spk', dest='source_utt2spk', help='The source utt2spk file to split by actual speaker (instead of emotion)', required=True)
	parser.add_argument('--output_dir', dest='output_dir', help='The output_dir where we\'ll write utt2spk-norm.', required=True)
	args = parser.parse_args()

	utterances = list()
	with open(args.source_utt2spk, 'r') as f:
		for line in f:
			utterance_id, _ = line.split(' ')
			utterances.append(utterance_id)

	utterances_by_speaker = defaultdict(list)
	for utterance in utterances:
		assert 'iemocap' in utterance
		session = utterance.split('-')[2][0:2]
		speaker_gender = utterance.split('-')[6]
		speaker = "%s-%s" % (session, speaker_gender)
		utterances_by_speaker[speaker].append(utterance)

	print("Found speakers: %s" % (utterances_by_speaker.keys()))

	with open(os.path.join(args.output_dir, 'utt2spk-norm'), 'w') as f:
		for speaker_id, speaker in enumerate(sorted(utterances_by_speaker.keys())):
			for utterance in utterances_by_speaker[speaker]:
				f.write('%s %s\n' % (utterance, speaker_id))
