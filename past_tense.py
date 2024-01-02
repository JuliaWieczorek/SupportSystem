import spacy



def past_tense_words(text):
   doc = nlp(text)
   past_tense_words = [token.text.lower() for token in doc if token.tag_ == "VBD"]  # VBD tag represents past tense verbs
   return past_tense_words

sentence = "I went to the store yesterday and bought some groceries."
past_words = past_tense_words(sentence)
print("Past tense words:", past_words)
