from indicnlp.tokenize import indic_tokenize

# Sample Gujarati text
gujarati_text = "આવા સંજોગોમાં, આપણે કઈ રીતે આગળ વધી શકીએ?"

# Tokenize the text
tokens = list(indic_tokenize.trivial_tokenize(gujarati_text))
print(tokens)
