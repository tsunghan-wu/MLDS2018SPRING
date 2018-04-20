from pattern.en import lemma
import numpy as np
word_collection = np.load('word_collection.npy').tolist()
print(word_collection)
import pickle
for word in word_collection:
	print word,lemma(word)
mirror = { word:lemma(word) for word in word_collection}
pickle.dump(mirror,open('mirror.pickle','wb'))