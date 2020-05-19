#!/usr/bin/env python

# author: aa4461

import re
import csv
import sys
import glob
import os.path
import numpy as np
from scipy.io import loadmat
from collections import Counter

COLLAPSED_EMOVOXCELEB_EMOTIONS = {
	'anger': 'anger/disgust',
	'disgust': 'anger/disgust',
	'contempt': 'anger/disgust',
	'fear': 'fear/surprise',
	'happiness': 'happiness',
	'neutral': 'neutral',
	'sadness': 'sadness',
	'surprise': 'fear/surprise'
}
# note that changing this set of emotions (via addition, removal or reordering)
# requires re-generating the IEMOCAP input data and re-training the final model!
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
EMOTION_TO_ID = {emotion: id for id, emotion in enumerate(EMOTIONS)}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

class UtteranceDetails:
	def __init__(self, src_file, dialogue_id, utterance_id, emotion):
		self.src_file = src_file
		self.utterance_id = utterance_id
		self.emotion = emotion

	def __repr__(self):
		return "%s: %s" % (self.get_id(), self.emotion)

	def get_id(self):
		return "%s-%s-%s-%s" % (
			self.emotion, self.src_file, self.utterance_id)

	def get_filename(self):
		return "utt%s.mp4" % (self.utterance_id)

def get_majority_vote_emotion(frame_labels):
	votes_by_emotion = Counter()
	for frame_label in frame_labels:
		emotion = EMOTIONS[np.argmax(frame_label)]
		collapsed_emotion = COLLAPSED_EMOVOXCELEB_EMOTIONS[emotion]
		votes_by_emotion[collapsed_emotion] += 1

	max_votes = -1
	max_emotion = None
	for emotion, votes in votes_by_emotion.items():
		if votes > max_votes:
			max_votes = votes
			max_emotion = emotion 

	return max_emotion

# file_info: A.J._Buckley/test/Y8hIVOBuels_0000001.wav
# actual_file_path: 
def get_wav_file(wav_files, file_info):
	wav_file_group, wav_file_id = file_info.split('/')[-1].split('.')[0].split('_')
	wav_file_key = (wav_file_group, int(wav_file_key))
	if wav_file_key not in wav_files:
		raise Exception("Unable to find wav_file %s, %s" % (file_info, wav_file_key))
	return wav_files[wav_file_key]

# note that the utt2spk file we generate actually each 
# utterance to a numerical encoding of the utterance's 
# emotion label according to MELD (so that we can lever 
# the existing vox2/run.sh with minimal modification)
def generate_utt2spk(utterances, output_data_dir):
	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			f.write("%s %s\n" % (utterance.get_id(), EMOTION_TO_ID[utterance.emotion]))

def generate_wavscp(utterances, input_data_dir, output_data_dir):
	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			mp4_file_path = os.path.join(input_data_dir, utterance.get_filename())
			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (mp4_file_path)
			f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))

EMOTION_LOGITS = 'corpora/voxceleb/vox1/senet50-ferplus-logits.mat'
def main():
	input_matlab = sys.argv[1]
	input_data_dir = sys.argv[2]
	labeling_mode = sys.argv[3]
	output_data_dir = sys.argv[4]

	# format = wav file -> frame -> emotion 
	matlab_file = loadmat(EMOTION_LOGITS)

	all_wav_logits = mat['wavLogits'][0]
	all_wav_info = mat['images'][0][0][0][0]

	assert len(all_wav_logits) == len(all_wav_info)

	wav_files = {}
	all_dev_files = set(glob.glob(os.path.join(input_data_dir, 'dev/wav/*/*/*.wav')))
	all_test_files = set(glob.glob(os.path.join(input_data_dir, 'test/wav/*/*/*.wav')))
	for wav_file in all_dev_files | all_test_files:
		# dev/wav/id10046/0gaIk-T8tcM/00001.wav
		wav_file_group = wav_file.split('/')[-2]
		wav_file_id = int(wav_file.split('/')[-1].split('.')[0])
		wav_file_key = (wav_file_group, wav_file_id)
		
		if wav_file_key in wav_files:
			raise Exception("Duplicate wav file found! %s" % (wav_file_key))
		wav_files[wav_file_key] = wav_file
	
	utterances = []
	for utterance_id in range(len(all_wav_logits)):
		wav_logits = all_wav_logits[utterance_id]
		wav_file = get_wav_file(wav_files, all_wav_info[utterance_id][0])

		if labeling_mode == 'majority':
			emotion = get_majority_vote_emotion(frame_annotations)

		utterances.append(Utterance(wav_file, utterance_id, emotion))

	generate_utt2spk(utterances, output_data_dir)
	generate_wavscp(utterances, input_data_dir, output_data_dir)

if __name__ == "__main__":
	main()
