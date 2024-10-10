import datasets
import numpy as np
import os
import faiss
from retrieval_index import RetrievalIndex
import torch

class DPRRetriever(RetrievalIndex):
  """
  Class for performing DPR using Faiss.
    @param corpus: A huggingface dataset with 'text' and potentially 'embedding' as keys.
    @param precomputed_index: A path to a precomputed index for a corpus. If None, the index will be created and saved in a file titled "amcn_index.faiss"
    @param similarity_func: The function to be used for comparing the similarity between embeddings. If you have created an index previously, this parameter has to correspond to the function used for that index! Options are ['L2_norm', 'cos_sim']
    @param device: The device to be used, options are: ['cpu', 'cuda']

  """
  def __init__(self, corpus: datasets.arrow_dataset.Dataset,
               precomputed_index: str,
               device):
    super().__init__(corpus)
    self.corpus = corpus.with_format('torch', device=device)
    self.device = device

    if precomputed_index is not None and os.path.exists(precomputed_index): # load index to corpus
      self.load_index(precomputed_index)
    else: # create a new index, save it and add it to the corpus
      self.build_new_index()
      self.load_index(f'data/retrieval/{self.index_savename}.faiss')

  def build_new_index(self):
    """
    Create a new faiss index based on the specifications in __init__
    """
    
    self.index_savename = 'passage_index'
    _corpus = self.corpus.with_format('torch', device=self.device)
    
    if 'embedding' not in self.corpus.features.keys(): # embed the dataset
      print(f'You must embed the corpus!')
      raise NotImplementedError()
    _corpus.add_faiss_index(column='embedding') #device=self.device.index
    
    self.custom_index = faiss.IndexFlatIP(_corpus['embedding'].size(1))
    embeddings = _corpus['embedding'].cpu().numpy()   
    self.custom_index.add(embeddings)
  
    self.save_index(f'{self.index_savename}.faiss') # save the newly created index

  def lookup(self,queries: list[str], queries_emb: np.ndarray, top_k: int = 100):
    """
    Function that retrieves the top_k most relevant passages with respect to the input queries.
    @param queries: numpy array of shape (n_sentences x embed_dim)
    @returns : the top_k passages and associated score for each input query
    """
    # encode query
    assert type(queries_emb) == np.ndarray, 'The query to lookup has to be an numpy array (embedding)!'
    import pdb
    with torch.no_grad():
        scores, passages = self.corpus.get_nearest_examples_batch('embedding', queries_emb, k=top_k) # retrieve top_k
    
    # unpack results
    result = {q: None for q in queries}
    for i in range(len(queries)): # loop over each queries
        top_100_scores = scores[i][:100]
        top_100_answers = passages[i]['answer'][:100]
        result[queries[i]] = (top_100_scores, top_100_answers)
    return result