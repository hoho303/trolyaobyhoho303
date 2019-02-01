from tokenization.crf_tokenizer import CrfTokenizer
from word_embedding.word2vec_gensim import Word2Vec
from text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict
# Please give the correct paths
# Load word2vec model from file. If you want to train your own model, please go to README or check word2vec_gensim.py
word2vec_model = Word2Vec.load('models/pretrained_word2vec.bin')

tokenizer = CrfTokenizer(config_root_path='tokenization/',
                         model_path='models/pretrained_tokenizer.crfsuite')
sym_dict = load_synonym_dict('data/sentiment/synonym.txt')
keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                    model_path='models/app.h5',
                                                    max_length=10, n_epochs=10,
                                                    sym_dict=sym_dict, n_class=3)
# Load and prepare data
X, y = keras_text_classifier.load_data()

# Train your classifier and test the model
keras_text_classifier.train(X, y)
label_dict = {0: 'mo_vnexpress', 1: 'mo_dantri', 2: 'mo_truyenfull'}
test_sentences = ['vnexpress come', 'dân trí lên', 'truyện đê', 'vnexpress chế']
labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
print(labels)



