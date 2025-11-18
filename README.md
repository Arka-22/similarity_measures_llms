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
Examples are extracted depth wise ( from 4 to 11 )
The proof chain is given in a cumulative manner to the model to get the last hidden state
The curvature similarities are calcualted for every pair and the mean is calcualted per depth level

To do : 
Implement scripts to pass arguments like model name through cmd prompt
Optimize and clean the code
