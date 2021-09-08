# This parser uses BERT transformer model to understand 
# what kind of application the user wants

from summarizer import Summarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = ["we", "I", "They", "our", "mine", "my", "want", 
"capture", "aim", "have", "has", "had", "tried", "try", "to", "use"]

class Textparser:

    model = Summarizer()

    def __init__(self, ratio, num_sentences):
        self.ratio = ratio
        self.num_sentences = num_sentences
    def extract(self, sentence):
        summary_out = self.model(sentence,num_sentences = self.num_sentences)
        return summary_out
    def elbow(self,body):
        res = self.model.calculate_elbow(body,k_max=10)
        return res
    def embeddings(self,body):
        summary_emb = self.model.run_embeddings(body)
        return summary_emb
    def keywording(self,body):
        n_gram_range = (3,3)
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=STOP_WORDS).fit([body])
        candidates = count.get_feature_names()
        candidates_str = ' '.join(map(str,candidates))
        body_embeddings = self.embeddings(body)
        candidate_embeddings = self.model.run_embeddings(candidates_str)
        top_n = 5
        distances = cosine_similarity(body_embeddings,candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        return keywords




