# similarity_measures_llms
Find how similar the logic flow of different LLMs are based on curvature similarities



Dataset used : 
deepseek_2k_RP_test.parquet

Models used : 

1) causality-grammar/bart-base-fullfinetuned_40k
2) Amartya77/flan-t5-base-cot-full-finetuned_40k_wor
3) causality-grammar/qwen3-1.7B-fullfinetuned

To check the similarity scores for 1 and 2 models run the t5.py file ( model and tokenizer )
To check the similarity scores for 3 model run the qwen.py file


Flow of the code :
1) Examples are extracted depth wise ( from 6 to 11 )
2) The proof chain is given in a cumulative manner to the model to get the last hidden state
3) The curvature similarities are calcualted for every pair and the mean is calcualted per depth level

To do : 
1) Implement scripts to pass arguments like model name through cmd prompt
2) Optimize and clean the code


Results :

| Depth | BART |  Flan-T5 |
| ----: | :--: | :------: |
|     6 | 0.73 | **0.80** |
|     7 | 0.71 | **0.77** |
|     8 | 0.67 | **0.74** |
|     9 | 0.64 | **0.72** |
|    10 | 0.62 | **0.70** |
|    11 | 0.62 | **0.70** |

