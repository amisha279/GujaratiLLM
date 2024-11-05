import random
from spacy.training import Example

# Load the trained model
nlp = spacy.load("gujarati_ner_model")

# Add new training data
NEW_TRAIN_DATA = [
    ("અમિત શાહ ગૃહ મંત્રાલયના વડા છે.", {"entities": [(0, 8, "PERSON")]}),
    ("ટાટા એક બહુ મોટી કંપની છે.", {"entities": [(0, 4, "ORG")]}),
]

# Get the NER component
ner = nlp.get_pipe("ner")

# Add labels if necessary
for _, annotations in NEW_TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Fine-tune the model with the new data
optimizer = nlp.resume_training()
for i in range(10):
    random.shuffle(NEW_TRAIN_DATA)
    losses = {}
    for text, annotations in NEW_TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], losses=losses, drop=0.35, sgd=optimizer)
    print(f"Losses at iteration {i}:", losses)

# Save the fine-tuned model
nlp.to_disk("fine_tuned_gujarati_ner_model")

# Test the fine-tuned model
nlp = spacy.load("fine_tuned_gujarati_ner_model")
doc = nlp("નરેન્દ્ર મોદી અને અમિત શાહ ભારત ના નેતાઓ છે.")
for ent in doc.ents:
    print(ent.text, ent.label_)
