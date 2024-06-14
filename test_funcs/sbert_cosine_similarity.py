from sentence_transformers import SentenceTransformer, util
sentences = ["I'm happy", "I'm full of happiness"]

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
#model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-cos-v5')
#model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
#model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
#model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
#model = SentenceTransformer('sentence-transformers/sentence-t5-large')
#model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
#model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
#model = SentenceTransformer('sentence-transformers/')
#model = SentenceTransformer('sentence-transformers/')

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

sim = util.pytorch_cos_sim(embedding_1, embedding_2)
print(sim)
