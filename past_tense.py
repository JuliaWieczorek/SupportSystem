import spacy

nlp = spacy.load("en_core_web_sm")

def focus_past_count(dialog):
   past_words = set()
   past_expressions = set()

   for i in dialog:
      for token in nlp(i):
         if token.tag_ in ["VBD", "VBN"]:  # VBD - Past tense, VBN - Past participle
            past_words.add(token.text.lower())
         if token.text.lower() in ["yesterday", "ago", "last", "past"]:
            past_expressions.add(token.text.lower())
   return len(past_words) + len(past_expressions)

sentence = ['Hello', 'I', 'am', 'having', 'a', 'lot', 'of', 'anxiety', 'about', 'quitting', 'my', 'current', 'job', '.', 'It', 'is', 'too', 'stressful', 'but', 'pays', 'well', 'I', 'have', 'to', 'deal', 'with', 'many', 'people', 'in', 'hard', 'financial', 'situations', 'and', 'it', 'is', 'upsetting', 'I', 'do', ',', 'but', 'often', 'they', 'are', 'not', 'going', 'to', 'get', 'back', 'to', 'what', 'they', 'want', '.', 'Many', 'people', 'are', 'going', 'to', 'lose', 'their', 'home', 'when', 'safeguards', 'are', 'lifted', 'That', 'is', 'true', 'but', 'sometimes', 'I', 'feel', 'like', 'I', 'should', 'put', 'my', 'feelings', 'and', 'health', 'first', 'Probably', 'not', '.', 'I', 'was', 'with', 'the', 'same', 'company', 'for', 'a', 'long', 'time', 'and', 'I', 'consistently', 'get', 'a', 'bonus', 'every', 'year', 'I', 'could', 'try', '.', 'It', 'mostly', 'gets', 'to', 'me', 'at', 'the', 'end', 'of', 'the', 'day', 'That', 'is', 'also', 'true', '.', 'Sometimes', 'I', 'wonder', 'if', 'it', 'really', 'is', 'for', 'me', 'though', 'That', 'is', 'true', '.']


past_words = focus_past_count(sentence)
print("Past tense words:", past_words)
