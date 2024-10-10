import datasets
import torch
import os
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder


def load_nq_dataset():
    """
    NOTE: The validation split argument passed corresponds to the "test" split reported by the DPR authors. 
          On huggingface it is stated that this dataset contains 3610 rows which aligns with table 1 of the DPR paper 
    """
    ds = datasets.load_dataset("google-research-datasets/nq_open", split='validation')
    assert ds.num_rows == 3610, 'The wrong dataset split has been loaded...'
    return ds

class EmbeddingModel:
    def __init__(self, corpus) -> None:
        self.corpus = self.unpack_corpus(corpus)
        self.__init_model() # initialize embedding models and tokenizers
    
    def __init_model(self):
        passage_modelname = "facebook/dpr-ctx_encoder-single-nq-base"

        self.passage_embedder = DPRContextEncoder.from_pretrained(passage_modelname)
        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_modelname)
        
    def unpack_corpus(self, corpus):
        # function for unpacking the corpus since questions contain multiple correct answers
        print("unpacking dataset..")
        D_unpacked = {'question': [], 'answer': []}
        for i in range(corpus.num_rows):
            for j in range(len(corpus[i]['answer'])):
                D_unpacked['question'].append(corpus[i]['question'])
                D_unpacked['answer'].append(corpus[i]['answer'][j])
            if i % 500 == 0:
                print(f'Iter: {i}/{corpus.num_rows}')
        return datasets.Dataset.from_dict(D_unpacked)
    
    def embed_function(self, batch):

        # loop over the possible answers
        p_emb = []
        for answers in batch['answer']:
            p_tokenized = self.passage_tokenizer(answers, padding=True, return_tensors='pt')
            emb = self.passage_embedder(**p_tokenized).pooler_output
            p_emb.append(emb.squeeze())
        
        return {'embedding': p_emb}

    def embed_corpus(self):
        print("Embedding corpus...")
        self.corpus = self.corpus.map(self.embed_function, batched=True)
    
    def save_corpus(self):
        if not os.path.exists(os.path.join(os.getcwd(), 'data')):
            os.mkdir(os.path.join(os.getcwd(), 'data'))
        self.corpus.save_to_disk("data/NQ_emb_corpus.hf")

if __name__ == '__main__':
    # load, embed and save the NQ dataset
    ds = load_nq_dataset()
    EM = EmbeddingModel(corpus=ds)
    EM.embed_corpus()
    EM.save_corpus()
    
        

            
        

