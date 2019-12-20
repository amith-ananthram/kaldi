#!/usr/bin/env python

import sys
import os.path
import scoring_utils

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'
PREDICTIONS_FILE = 'predictions.ark'

#def generate_utt2spk(utterances, output_data_dir):
#	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
#		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
#			f.write("%s %s\n" % (utterance.get_id(), EMOTION_TO_ID[utterance.emotion]))

#def generate_wavscp(utterances, input_data_dir, output_data_dir):
#	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
#		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
#			mp4_file_path = os.path.join(input_data_dir, utterance.get_filename())
#			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (mp4_file_path)
#			f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))

def get_wav_labels_and_predictions(input_dir, prefix):
	utts_to_wav_creation = {}
	with open(os.path.join(input_dir, '%s_%s' % (prefix, WAV_FILE)), 'r') as f:
		for line in f:
			(utt, wav_creation) = line.split(maxsplit=1)
			if utt in utts_to_wav_creation:
				raise Exception('%s duped in %s wav file' % (utt, prefix))
			utts_to_wav_creation[utt] = wav_creation

	utts_to_labels = {}
	with open(os.path.join(input_dir, '%s_%s' % (prefix, UTT2SPK_FILE)), 'r') as f:
		for line in f:
			(utt, label) = line.split(maxsplit=1)
			if utt in utts_to_labels:
				raise Exception('%s duped in %s utt2spk file' % (utt, prefix))
			utts_to_labels[utt] = label

	utts_to_predictions = scoring_utils.parse_predictions_ark('%s_%s' % (prefix, PREDICTIONS_FILE))

	return (utts_to_wav_creation, utts_to_labels, utts_to_predictions)

def main():
	input_data_dir = sys.argv[1]
	output_data_dir = sys.argv[2]

	meld_wav, meld_labels, meld_predictions = get_wav_labels_and_predictions(input_data_dir, 'meld')
	iemocap_wav, iemocap_labels, iemocap_predictions = get_wav_labels_and_predictions(input_data_dir, 'iemocap')

	print(meld_wav)
	print(meld_labels)
	print(meld_predictions)
	print(iemocap_wav)
	print(iemocap_labels)
	print(iemocap_predictions)

	# get all correctly labeled IEMOCAP examples
	# get all correctly labeled MELD examples
	# generate new utterance ids for them (for all utterances), utt2spk accordingly
	# use the filepath in wav.scp for all of them

if __name__ == "__main__":
	main()