from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import QuickText
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
documents = ["I write a book and eat a cake.", "I am really happy today because I met with Tom."]
sp = WhiteSpacePreprocessing(documents)
preprocessed_docs, unpreprocessed_doc, vocab = sp.preprocess()

qt = QuickText("bert-base-nli-mean-tokens",
                text_for_bert=preprocessed_docs,
                text_for_bow=unpreprocessed_doc)

training_dataset = qt.load_dataset()

ctm = CombinedTM(input_size=len(qt.vocab), bert_input_size=768, n_components=2)

ctm.fit(training_dataset) # run the model

print(ctm.get_topics())
