
import pdb
from data_utils import load_nq_dataset
from dpr_retriever import DPRRetriever
import torch
import datasets
from itertools import batched

# class DPREvaluator:
#     def __init__(self, dataset_name: str = 'NQ') -> None:
#         self.model = DPRRetriever(corpus=dataset2func[dataset_name](), precomputed_index=None, similarity_func='IP')


from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder

        

def main(corpus_path: str = 'data/NQ_emb_corpus.hf', 
         precomputed_index: str = 'data/retrieval/passage_index.faiss'):
    
    ds = load_nq_dataset()
    corpus = datasets.load_from_disk(corpus_path) # load embedded corpus
    
    dpr = DPRRetriever(corpus, precomputed_index, device='cpu')
    
    question_modelname = "facebook/dpr-question_encoder-single-nq-base"
    question_embedder = DPRQuestionEncoder.from_pretrained(question_modelname)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_modelname)
    
    all_questions = ds['question']
    batch_size = 4
    for i, questions in enumerate(batched(all_questions, batch_size)):
        
        # embed questions
        queries = question_embedder(**question_tokenizer(questions, 
                                                         padding=True, 
                                                         return_tensors='pt')).pooler_output
        # calculate top-100 most similar answers
        result_dict = dpr.lookup(questions, queries.detach().numpy())
        
        top_100_correct_count = 0
        top_20_correct_count = 0
        # loop over the answers to each question
        for j, answers in enumerate(ds['answer'][i : min(i+batch_size, ds.num_rows)]):
            
            predictions = result_dict[questions[j]][1] # extract top 100 predictions
            
            has_top_100_count = False
            for answer in answers:
                
                if answer in predictions[:20]:
                    top_20_correct_count += 1
                    top_100_correct_count = top_100_correct_count + 1 if not has_top_100_count else top_100_correct_count
                    break                     
                elif answer in predictions:
                    top_100_correct_count += 1
                    has_top_100_count = True        
                        
        
        top_100_acc = top_100_correct_count / ds.num_rows
        top_20_acc = top_20_correct_count / ds.num_rows
        print(f'top 100 acc: {top_100_acc}')
        print(f'top 20 acc: {top_20_acc}')
        

if __name__ == '__main__':
    main()
    