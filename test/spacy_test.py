import spacy

nlp = spacy.load('vi_spacy_model')
doc = nlp('xin chào, tôi tên là A')
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)