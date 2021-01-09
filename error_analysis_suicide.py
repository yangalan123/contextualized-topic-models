from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import QuickText
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import pickle, json
import numpy as np
N_component = 10
with open("aligned_ds_details.pkl", "rb") as f_in:
    data = pickle.load(f_in)
    documents = []
    profiles = []
    for item in data:
        profiles.append({
            "model_input_range": len(item["model_input"]),
            "pred": item["pred"],
            "label": item["label"],
            "dataset_type": item["dataset_type"]
        })
        documents.extend(item["model_input"])

sp = WhiteSpacePreprocessing(documents)
preprocessed_docs, unpreprocessed_doc, vocab = sp.preprocess()

qt = QuickText("bert-base-nli-mean-tokens",
               text_for_bert=preprocessed_docs,
               text_for_bow=unpreprocessed_doc)

training_dataset = qt.load_dataset()

ctm = CombinedTM(input_size=len(qt.vocab), bert_input_size=768, n_components=10)

ctm.fit(training_dataset) # run the model

ctm.save("models/ctm.pth")

print("------Topic-Word Distribution-------")
print(ctm.get_topics())
doc_topic_distribution = ctm.get_doc_topic_distribution(training_dataset)
user_dist = dict()
start = 0
label_user_dict = dict()
pred_user_dict = dict()
confusion_user_dict = dict()
for i, item in enumerate(profiles):
    user_dist[i] = np.mean([doc_topic_distribution[start: start + item["model_input_range"]]])
    label, pred = item["label"], item["pred"]
    if label != pred:
        key = (label, pred)
        if key not in confusion_user_dict:
            confusion_user_dict[key] = list()
        confusion_user_dict[key].append(i)
    if label not in label_user_dict:
        label_user_dict[label] = list()
    if pred not in pred_user_dict:
        pred_user_dict[pred] = list()
    label_user_dict[label].append(i)
    pred_user_dict[pred].append(i)

print("------Category-Topic Distribution (True Label)-------")
label_topic_dist = []
for i in range(4):
    label_topic_dist.append(np.mean([user_dist[x] for x in label_user_dict[i]]))
print(np.stack(label_topic_dist, axis=0))

print("------Category-Topic Distribution (Pred Label)-------")
pred_topic_dist = []
for i in range(4):
    pred_topic_dist.append(np.mean([user_dist[x] for x in pred_user_dict[i]]))
print(np.stack(pred_topic_dist, axis=0))

print("------Category-Topic Distribution (Confusion)-------")
for key in confusion_user_dict:
    # confusion_topic_dist.append(np.mean([user_dist[x] for x in confusion_user_dict[i]]))
    _key = "{} (label) -> {} (pred)".format(key[0], key[1])
    _item = {
        "confusion_type": _key,
        "label_dist": label_topic_dist[key[0]],
        "pred_dist": pred_topic_dist[key[1]]
    }
    print(json.dumps(_item, indent=4))






