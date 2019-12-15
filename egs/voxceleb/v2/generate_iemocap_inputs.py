#!/usr/bin/env python

import re
import csv
import sys
import os.path

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

class UtteranceDetails:
	def __init__(self, session, dialogue_id, utterance_id, emotion):
		self.session = session
		self.
		self.src_file = src_file
		self.dialogue_id = dialogue_id
		self.utterance_id = utterance_id
		self.emotion = emotion

	def __repr__(self):
		return "%s: %s" % (self.get_id(), self.emotion)

	def get_id(self):
		return "%s-%s-%s-%s" % (
			self.emotion, self.src_file, self.dialogue_id, self.utterance_id)

	# Ses01F_impro01/Ses01F_impro01_F000.wav
	# Sr.No,Utterance,Speaker,Emotion,Session_Number,Mocap_Source,Dialogue_Type,Dialogue_Number,Speaker,Utterance_Number,StartTime,EndTime,Emotion_Label
	# 1303,Excuse me.,F,neu,01,_,improvisation,01,F,000,6.2901,8.2357,3
	def get_filename(self):
		return "dia%s_utt%s.mp4" % (self.dialogue_id, self.utterance_id)

def get_utterances(input_csv):
	utterances = []
	with open(input_csv) as f:
		csv_reader = csv.reader(f, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
				continue
			else:
				speaker = row[2]
				session_number = row[4]
				dialogue_type = row[6]
				dialogue_number = row[7]
				addressee = row[8]
				utterance_number = row[9]
				emotion_label = row[12]
				utterances.append(
					UtteranceDetails(
						speaker, 
						session_number, 
						dialogue_type, 
						dialogue_number,
						addressee,
						utterance_number,
						emotion_label
					)
				)
				line_count += 1
	return utterances

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

def main():
	session = sys.argv[1]
	input_csv = sys.argv[2]
	input_data_dir = sys.argv[3]
	output_data_dir = sys.argv[4]

	utterances = get_utterances(input_csv)

	generate_utt2spk(utterances, output_data_dir)
	generate_wavscp(utterances, input_data_dir, output_data_dir)

if __name__ == "__main__":
	main()