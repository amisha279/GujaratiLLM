import random
import spacy
from spacy.training import Example

# Create a blank Gujarati model
nlp = spacy.blank("xx")

# Add the NER component
ner = nlp.add_pipe("ner")

# Define labels
ner.add_label("PERSON")
ner.add_label("ORG")
ner.add_label("GPE")

# Training data
TRAIN_DATA = [
    ("આમિત શાહ ભારત ના ગૃહમંત્રી છે.", {"entities": [(0, 9, "PERSON")]}),
    ("ગૂગલ એ એક મોટી કંપની છે.", {"entities": [(0, 5, "ORG")]}),
]

# Train the NER model
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], losses=losses, drop=0.5, sgd=optimizer)
    print(losses)

# Save the model
nlp.to_disk("gujarati_ner_model")

# Load and test the model
nlp = spacy.load("gujarati_ner_model")
doc = nlp("નરેન્દ્ર મોદી ભારત ના પ્રધાનમંત્રી છે.")
for ent in doc.ents:
    print(ent.text, ent.label_)
