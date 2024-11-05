import spacy
import indicnlp.tokenize
from spacy.tokens import Doc
from spacy.language import Language

# Load spaCy
nlp = spacy.blank('xx')  # Create a blank model

# Custom tokenizer
@Language.component("gujarati_tokenizer")
def gujarati_tokenizer(doc):
    words = list(indicnlp.tokenize.indic_tokenize.trivial_tokenize(doc.text))
    return Doc(nlp.vocab, words=words)

# Add the custom tokenizer to the pipeline
nlp.add_pipe("gujarati_tokenizer", first=True)

# Sample text
doc = nlp("આવા સંજોગોમાં, આપણે કઈ રીતે આગળ વધી શકીએ?")
print([token.text for token in doc])
