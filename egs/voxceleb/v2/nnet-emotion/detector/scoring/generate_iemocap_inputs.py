#!/usr/bin/env python

# author: aa4461

import os
import csv
import glob
import argparse
import numpy as np
from bidict import bidict

IEMOCAP_DIR = 'corpora/iemocap/'
IEMOCAP_WAV_DIR = os.path.join(IEMOCAP_DIR, 'IEMOCAP_full_release')
IEMOCAP_INPUT_CSV_DIR = os.path.join(IEMOCAP_DIR, 'csvs')

IEMOCAP_EMOTIONS = bidict({emo: emo for emo in ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur']})
IEMOCAP_DIALOGUE_TYPES = {'improv': ['improvisation'], 'script': ['script'], 'all': ['improvisation', 'script']}

SUPPORTED_TARGET_EMOTION_MODES = {'select', 'collapse'}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

# TODO: share between generate_iemocap_inputs and generate_corpora_inputs
class IemocapUtteranceDetails:
	def __init__(self, session, mocap_source, dialogue_type, 
		dialogue_number, utterance_number, utterance, speaker,
		src_file, orig_emotion, mapped_emotion):
		self.session = session
		self.mocap_source = mocap_source
		self.dialogue_type = dialogue_type
		self.dialogue_number = dialogue_number
		self.utterance_number = utterance_number
		self.utterance = utterance
		self.speaker = speaker
		self.src_file = src_file
		self.orig_emotion = orig_emotion
		self.mapped_emotion = mapped_emotion

	# old id format (required to find WAV files)
	def get_old_id(self):
		return "Ses%s%s_%s%s_%s%s" % (
			self.session,
			self.mocap_source,
			self.get_dialogue_type_shortname(),
			self.dialogue_number,
			self.speaker,
			self.utterance_number
		)

	# new id format (required for LDA / PLDA training)
	def get_id(self):
		return "%s-%s-%s%s-%s-%s-%s-%s-iemocap" % (
			self.mapped_emotion, self.src_file, self.session, self.mocap_source, 
			self.dialogue_type, self.dialogue_number, self.utterance_number, self.speaker)

	def get_emotion(self):
		return self.mapped_emotion

	def get_dialogue_type_shortname(self):
		if self.dialogue_type == "improvisation":
			return "impro"
		elif self.dialogue_type == "script":
			return "script"
		else:
			raise Exception("Unsupported dialogue type: %s" % (self.dialogue_type))

	# example filename: Ses01F_impro01/Ses01F_impro01_F000.wav
	def get_wav_filename(self):
		wav_dir = os.path.join(IEMOCAP_WAV_DIR, 'Session%s/sentences/wav' % (int(self.session)))
		return os.path.join(wav_dir, "Ses%s%s_%s%s/%s.wav" % (
			self.session,
			self.mocap_source,
			self.get_dialogue_type_shortname(),
			self.dialogue_number,
			self.get_old_id()
		))

# TODO: share between generate_iemocap_inputs and generate_corpora_inputs
def get_iemocap_utterances(subsets, config, emotion_mapper):
	utterances = []
	for subset in subsets:
		input_csv = os.path.join(IEMOCAP_INPUT_CSV_DIR, 'Session%s.csv' % (subset))

		src_file = None
		if input_csv.find("train") != -1:
			src_file = "train"
		elif input_csv.find("dev") != -1:
			src_file = "dev"
		else:
			src_file = "test"

		with open(input_csv) as f:
			csv_reader = csv.reader(f, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0:
					line_count += 1
					continue
				else:
					# example CSV row
					# 0 Sr.No: 1303
					# 1 Session_Number: 01
					# 2 Mocap_Source: F
					# 3 Dialogue_Type: improvisation
					# 4 Dialogue_Number: 01
					# 5 Utterance_Number: 000
					# 6 StartTime: 6.2901
					# 7 EndTime: 8.2357
					# 8 Utterance: Excuse me.
					# 9 Speaker: F
					# 10 Emotion: neu
					session = row[1]
					mocap_source = row[2]
					dialogue_type = row[3]
					dialogue_number = row[4]
					utterance_number = row[5]
					utterance = row[8]
					speaker = row[9]
					orig_emotion = row[10]

					if int(session) != int(subset):
						raise Exception("Unexpectedly found %s in CSV for %s" % (session, subset)) 

					if config != 'all' and dialogue_type != IEMOCAP_DIALOGUE_TYPES[config]:
						continue

					if orig_emotion not in emotion_mapper:
						continue

					mapped_emotion = emotion_mapper[orig_emotion]
					utterances.append(
						IemocapUtteranceDetails(
							session,
							mocap_source,
							dialogue_type,
							dialogue_number,
							utterance_number,
							utterance,
							speaker,
							src_file,
							orig_emotion,
							mapped_emotion
						)
					)
					line_count += 1
	return utterances

# TODO: share between generate_iemocap_inputs and generate_corpora_inputs
def generate_utt2spk(emotion_mapper, utterances, output_data_dir):
	emotions_to_id = {emotion:idx for idx, emotion in enumerate(sorted(set(emotion_mapper.values())))}
	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			f.write("%s %s\n" % (utterance.get_id(), emotions_to_id[utterance.get_emotion()]))

# TODO: share between generate_iemocap_inputs and generate_corpora_inputs
def generate_wavscp(utterances, output_data_dir):
	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			if not os.path.exists(utterance.get_wav_filename()):
				raise Exception('Missing file: %s' % (utterance.get_wav_filename()))
			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (utterance.get_wav_filename())
			f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Kaldi input files based on specified train corpora and target configs.')
	parser.add_argument('--test_corpus', dest='test_corpus', help='The test corpus to use (iemocap{1|2|3|4|5})', required=True)
	parser.add_argument('--test_corpus_config', dest='test_corpus_config', help='How to select examples from the test corpus.', required=True)
	parser.add_argument('--target_emotions_mode', dest='target_emotions_mode', help='\'select\' or \'collapse\'', required=True)
	parser.add_argument('--target_emotions_config', dest='target_emotions_config', help='How to select or collapse labels.', required=True)
	parser.add_argument('--output_dir', dest='output_dir', help='Where to place generated utt2spk and wav.scp files', required=True)
	args = parser.parse_args()

	if 'iemocap' not in args.test_corpus:
		raise Exception("Unsupported test_corpus: %s" % (test_corpus))

	target_emotions_mode = args.target_emotions_mode
	if target_emotions_mode not in SUPPORTED_TARGET_EMOTION_MODES:
		raise Exception("Unsupported target_emotions_mode")

	emotion_mapper = {}
	target_emotion_config = args.target_emotions_config.split(',')
	for config_element in target_emotion_config:
		if '/' in config_element:
			emotions = config_element.split('/')
		else:
			emotions = [config_element]
		for emotion in emotions:
			assert emotion in IEMOCAP_EMOTIONS

		if target_emotions_mode == 'select':
			emotion_mapper[emotions[0]] = emotions[0]
		else:
			collapsed_label = '/'.join(emotions)
			for emotion in emotions:
				emotion_mapper[emotion] = collapsed_label

	utterances = get_iemocap_utterances([int(args.test_corpus[-1:])], args.test_corpus_config, emotion_mapper)

	generate_utt2spk(emotion_mapper, utterances, args.output_dir)
	generate_wavscp(utterances, args.output_dir)
