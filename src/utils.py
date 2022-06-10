
def max_word_count(df):
	max_words = 0
	for text in df['#1 String']:
		max_words = max(max_words, len(text.split()))
	for text in df['#2 String']:
		max_words = max(max_words, len(text.split()))
	
	return max_words
