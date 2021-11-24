# 运行命令参考

# WG4Rec on Sogou dataset
'python main.py --model_name WG4RecSogou --dataset sogou --metrics ndcg@5,ndcg@10,hit@5,hit@10 --test_sample_n 99 --lr 0.001 --l2 1e-4 --sent_max 10 --sample_pop 100 --regenerate 0 --gpu 0'

# WG4Rec on Adressa-1w dataset
'python main.py --model_name WG4RecAdressa --dataset adressa-1w --metrics auc --test_sample_n 1 --lr 1e-6 --l2 1e-2 --l2_bias 1 --dropout 0.5 --sent_max 50 --sample_pop 10 --regenerate 0 --gpu 0'
