import datasets
import os
import faiss


class RetrievalIndex:
  """
  Base class for performing retrieval.
  @param corpus: A huggingface dataset with 'embedding' as keys and 'passage_id' as values.
  """
  def __init__(self, corpus: datasets.arrow_dataset.Dataset):
    self.corpus = corpus
    self.base_path = os.path.join(os.getcwd(), 'data', 'retrieval')
    self.custom_index = None
    if not os.path.exists(self.base_path):
        os.makedirs(self.base_path, exist_ok=True)

  def __getitem__(self, item):
    return [i for i in self.corpus.select([item])]

  def save_index(self, file: str):
    """
    Function that saves the index to a file using `faiss.write_index`
    @param file: The filename or path where the index should be saved
    """
    path = os.path.join(self.base_path, file)
    faiss.write_index(self.custom_index, path)
    
  def load_index(self, file):
    """
    Function that loads an index using 'load_faiss_index'
    @param file: The filename or path to the precomputed index
    """
    assert os.path.exists(file), f'The index file {file} does not exist!'
    self.corpus.load_faiss_index('p_embedding', file)
    self.corpus.with_format('torch', device=self.device)



