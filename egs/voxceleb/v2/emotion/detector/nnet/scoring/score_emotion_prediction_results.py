#!/usr/bin/env python

import sys

import scoring_utils as su

def main():
	experiment_name = sys.argv[1]
	utt2labels_filepath = sys.argv[2]
	predictions_ark_filepath = sys.argv[3]
	output_path = sys.argv[4]

	utt2labels = su.parse_utt2labels(utt2labels_filepath)
	utt2predictions = su.parse_predictions_ark(predictions_ark_filepath)

	scores_using_most_frequent_emotion = su.score(
		'%s_most_freq' % (experiment_name), utt2labels, utt2predictions, su.get_most_frequent_emotion)
	scores_using_most_likely_emotion = su.score(
		'%s_most_likely' % (experiment_name), utt2labels, utt2predictions, su.get_most_likely_emotion)
	scores_using_longest_streak_emotion = su.score(
		'%s_longest_streak' % (experiment_name), utt2labels, utt2predictions, su.get_longest_streak_emotion)

	scores_using_most_frequent_emotion.print_results()
	scores_using_most_likely_emotion.print_results()
	scores_using_longest_streak_emotion.print_results()

	scores_using_most_likely_emotion.save_results(output_path)
	scores_using_most_frequent_emotion.save_results(output_path)
	scores_using_longest_streak_emotion.save_results(output_path)

if __name__ == "__main__":
	main()
