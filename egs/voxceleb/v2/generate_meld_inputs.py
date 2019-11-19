#!/usr/bin/env python

import csv
import sys
import os.path

# note that changing this set of emotions (via addition, removal or reordering)
# requires re-generating the MELD input data and re-training the final model!
EMOTION_TO_ID = {emotion: id for id, emotion in 
	enumerate(['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

class UtteranceDetails:
	def __init__(self, dialogue_id, utterance_id, emotion):
		self.dialogue_id = dialogue_id
		self.utterance_id = utterance_id
		self.emotion = emotion

	def __repr__(self):
		return "%s: %s" % (self.get_id(), self.emotion)

	def get_id(self):
		return "%s-%s" % (self.dialogue_id, self.utterance_id)

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
				utterances.append(UtteranceDetails(row[5], row[6], EMOTION_TO_ID[row[3]]))
				line_count += 1
	return utterances

# note that the utt2spk file we generate actually each 
# utterance to a numerical encoding of the utterance's 
# emotion label according to MELD (so that we can lever 
# the existing vox2/run.sh with minimal modification)
def generate_utt2spk(utterances, output_data_dir):
	with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
		for utterance in utterances:
			f.write("%s %s\n" % (utterance.get_id(), utterance.emotion))

def generate_wavscp(utterances, input_data_dir, output_data_dir):
	with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
		for utterance in utterances:
			mp4_file_path = os.path.join(input_data_dir, utterance.get_filename())
			wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -acodec pcm_s16le -|" % (mp4_file_path)
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