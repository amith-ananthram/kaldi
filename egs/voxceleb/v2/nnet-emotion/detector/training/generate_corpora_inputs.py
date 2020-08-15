#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
from bidict import bidict

CREMAD_DIR = 'corpora/CREMA-D'
CREMAD_WAV_DIR = os.path.join(CREMAD_DIR, 'AudioWAV')
CREMAD_INPUT_CSV = os.path.join(CREMAD_DIR, 'processedResults/summaryTable.csv')

CREMAD_EMOTIONS = bidict({'ang': 'A', 'dis': 'D', 'fea': 'F', 'hap': 'H', 'neu': 'N', 'sad': 'S'})
CREMAD_LABEL_TYPES = {'voice': 'Voice', 'multi': 'MultiModal'}
CREMAD_LABEL_AGREEMENTS = {'any', 'all'}

IEMOCAP_DIR = 'corpora/iemocap/'
IEMOCAP_WAV_DIR = os.path.join(IEMOCAP_DIR, 'IEMOCAP_full_release')
IEMOCAP_INPUT_CSV_DIR = os.path.join(IEMOCAP_DIR, 'csvs')

IEMOCAP_EMOTIONS = bidict({emo: emo for emo in ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur']})
IEMOCAP_DIALOGUE_TYPES = {'improv': ['improvisation'], 'script': ['script'], 'all': ['improvisation', 'script']}

SUPPORTED_CORPORA = {'cremad', *['iemocap%s' % subset for subset in range(1, 6)]}
SUPPORTED_TARGET_EMOTION_MODES = {'select', 'collapse'}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

class CremaDUtteranceDetails:
	def __init__(self, file_name, file_path, emotion):
		self.file_name = file_name
		self.file_path = file_path
		self.emotion = emotion

	def __repr__(self):
		return "%s: %s" % (self.file_name, self.emotion)

	def get_id(self):
		return "%s-%s" % (self.emotion, self.file_name)

	def get_emotion(self):
		return self.emotion

	def get_wav_filename(self):
		return self.file_path

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

	# new id format (required for LDA / PLDA training)
	def get_id(self):
		return "%s-%s-%s%s-%s-%s-%s-%s" % (
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
			self.get_old_id()[2:]
		))

def get_cremad_utterances(config, emotion_mapper):
	label_type, label_agreement = config.split('/')

	assert label_type in CREMAD_LABEL_TYPES
	assert label_agreement in CREMAD_LABEL_AGREEMENTS

	wav_filepaths = {w.split('/')[-1].split('.')[0]:w \
		for w in glob.glob(os.path.join(CREMAD_WAV_DIR, '*.wav'))}

	utterances = []
	with open(input_csv) as f:
		csv_reader = csv.DictReader(f, delimiter=',')
		for row in csv_reader:
			file_name = row["FileName"]
			wav_filepath = wav_filepaths[file_name]

			label_col_prefix = CREMAD_LABEL_TYPES[label_type]
			votes = row["%sVote" % label_col_prefix].split(":")
			levels = list(map(lambda l: float(l), row["%sLevel" % label_col_prefix].split(":")))

			if label_agreement == 'all' and len(votes) > 1:
				continue

			emotion = CREMAD_EMOTIONS.inverse[votes[np.argmax(levels)]]
			if emotion in emotion_mapper:
				utterances.append(CremaDUtteranceDetails(file_name, wav_filepath, emotion))
	return utterances

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
						UtteranceDetails(
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

def generate_utt2spk(emotion_mapper, utterances, output_data_dir):
	emotions_to_id = {emotion:idx for idx, emotion in enumerate(sorted(emotion_mapper.values()))}
	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			f.write("%s %s\n" % (utterance.get_id(), emotions_to_id[utterance.get_emotion()]))

def generate_wavscp(utterances, input_data_dir, output_data_dir):
	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			if not os.path.exists(utterance.get_wav_filename()):
				raise Exception('Missing file: %s' % (utterance.get_filename()))
			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (wav_file_path)
			f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Kaldi input files based on specified train corpora and target configs.')
	parser.add_argument('--train_corpora', dest='train_corpora', help='The training corpora to use (cremad, iemocap{1|2|3|4|5})', required=True)
	parser.add_argument('--train_corpora_config', dest='train_corpora_config', help='How to select examples from each training corpus.', required=True)
	parser.add_argument('--target_emotions_mode', dest='target_emotions_mode', help='\'select\' or \'collapse\'', required=True)
	parser.add_argument('--target_emotions_config', dest='target_emotions_config', help='How to select or collapse labels.', required=True)
	parser.add_argument('--output_dir', dest='output_dir', help='Where to place generated utt2spk and wav.scp files', required=True)
	args = parser.parse_args()

	train_corpora = set(args.train_corpora.split(','))
	if len(train_corpora - SUPPORTED_CORPORA) > 0:
		raise Exception("Unsupported train_corpora: %s" % (train_corpora - SUPPORTED_CORPORA))

	train_corpora_config = list(args.train_corpora_config.split(','))
	if len(train_corpora) != len(train_corpora_config):
		raise Exception("train_corpora / train_corpora_config length mismatch!")

	iemocap_to_include = set()
	iemocap_to_exclude = list(filter(lambda corpus: 'iemocap' in corpus, train_corpora))
	if len(iemocap_to_exclude) > 1:
		raise Exception("Please specify the part of IEMOCAP you want to exclude!")
	else:
		for i in range(1, 6):
			if 'iemocap%s' % i in iemocap_to_exclude:
				continue
			iemocap_to_include.add(i)

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
			for train_corpus in train_corpora:
				if train_corpus == 'cremad':
					assert emotion in CREMAD_EMOTIONS
				else:
					assert emotion in IEMOCAP_EMOTIONS

		if target_emotions_mode == 'select':
			emotion_mapper[emotions[0]] = emotions[0]
		else:
			collapsed_label = '/'.join(emotions)
			for emotion in emotions:
				emotion_mapper[emotion] = collapsed_label

	utterances = []
	for train_corpus, train_corpus_config in zip(train_corpora, train_corpora_config):
		if train_corpus == 'cremad':
			utterances.extend(get_cremad_utterances(train_corpus_config, emotion_mapper))
		else:
			utterances.extend(get_iemocap_utterances(iemocap_to_include, train_corpus_config, emotion_mapper))

	generate_utt2spk(utterances, args.output_dir)
	generate_wavscp(utterances, args.output_dir)


