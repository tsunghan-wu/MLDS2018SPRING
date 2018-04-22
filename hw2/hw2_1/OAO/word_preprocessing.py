from pattern.en import lemma
import numpy as np
word_collection = np.load('word_collection.npy').tolist()
print(word_collection)
import pickle
exception_list = {
	'tomatoes' : 'tomato',
	"onion's"  : 'onion',
	"parrot's"  : 'parrot',


}
for word in word_collection:
	print word,lemma(word)

mirror = { word:lemma(word) for word in word_collection}
mirror.update(exception_list)

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print(nltk.pos_tag(['man','men','women','April' , 'be','two']))

exit()
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

for word in mirror:
	if word != mirror[word] and mirror[word] != lemmatizer.lemmatize(word).lower():
		print(word,":",mirror[word] ,'vs:' ,  lemmatizer.lemmatize(word) , 'tag:' , nltk.pos_tag([word])[0][1])
pickle.dump(mirror,open('mirror.pickle','wb'))