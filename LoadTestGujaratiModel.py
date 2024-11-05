import spacy

# Load the trained model
nlp = spacy.load("gujarati_ner_model")

# Test the model with a sample text
test_text = "નરેન્દ્ર મોદી ભારત ના પ્રધાનમંત્રી છે."
doc = nlp(test_text)

# Print the entities found by the model
for ent in doc.ents:
    print(ent.text, ent.label_)
