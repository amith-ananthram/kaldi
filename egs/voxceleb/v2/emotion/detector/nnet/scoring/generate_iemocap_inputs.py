#!/usr/bin/env python

import re
import csv
import sys
import os.path

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

class UtteranceDetails:
	def __init__(self, session, mocap_source, dialogue_type, 
		dialogue_number, utterance_number, utterance, speaker, emotion_label):
		self.session = session
		self.mocap_source = mocap_source
		self.dialogue_type = dialogue_type
		self.dialogue_number = dialogue_number
		self.utterance_number = utterance_number
		self.utterance = utterance
		self.speaker = speaker
		self.emotion_label = emotion_label

	def __repr__(self):
		return "%s: %s" % (self.get_filename(), self.utterance)

	def get_id(self):
		return "%s_Ses%s%s_%s%s_%s%s" % (
			self.emotion_label,
                        self.session,
			self.mocap_source,
			self.get_dialogue_type_shortname(),
			self.dialogue_number,
			self.speaker,
			self.utterance_number
		)

	def get_dialogue_type_shortname(self):
		if self.dialogue_type == "improvisation":
			return "impro"
		elif self.dialogue_type == "script":
			return "script"
		else:
			raise Exception("Unsupported dialogue type: %s" % (self.dialogue_type))

	# example filename: Ses01F_impro01/Ses01F_impro01_F000.wav
	def get_filename(self):
		return "Ses%s%s_%s%s/%s.wav" % (
			self.session,
			self.mocap_source,
			self.get_dialogue_type_shortname(),
			self.dialogue_number,
                        self.get_id()[2:]
		)

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
				# 11 Emotion_Label: 3
				session = row[1]
				mocap_source = row[2]
				dialogue_type = row[3]
				dialogue_number = row[4]
				utterance_number = row[5]
				utterance = row[8]
				speaker = row[9]
				emotion_label = row[11]
				utterances.append(
					UtteranceDetails(
						session,
						mocap_source,
						dialogue_type,
						dialogue_number,
						utterance_number,
						utterance,
						speaker,
						emotion_label,
					)
				)
				line_count += 1
	return utterances

# note that in the utt2spk file we generate we map each 
# utterance to a numerical encoding of the utterance's 
# emotion label according to MELD (so that we can lever 
# the existing vox2/run.sh with minimal modification)
def generate_utt2spk(utterances, output_data_dir):
	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			f.write("%s %s\n" % (utterance.get_id(), utterance.emotion_label))

def generate_wavscp(utterances, input_data_dir, output_data_dir):
	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
		for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
			wav_file_path = os.path.join(input_data_dir, utterance.get_filename())
                        if not os.path.exists(wav_file_path):
                            raise Exception('Missing file: %s' % (utterance.get_id()))
			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (wav_file_path)
			f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))

def main():
	input_csv = sys.argv[1]
	input_data_dir = sys.argv[2]
	output_data_dir = sys.argv[3]

	utterances = get_utterances(input_csv)

	generate_utt2spk(utterances, output_data_dir)
	generate_wavscp(utterances, input_data_dir, output_data_dir)

if __name__ == "__main__":
	main()
