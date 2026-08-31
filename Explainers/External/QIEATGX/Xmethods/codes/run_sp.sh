#!/bin/bash
dataset=wikipedia # wikipedia, reddit
model=tgat # tgat, tgn
val=0.1 # sparsity_ratio
factual=1 # type of explanation, factual: 1, counterfactual: 0
existed=1 # target event existance, exist: 1, nonexist: 0


  ### pg_explainer_tg ###
# python QIEA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
# python GA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
# python GreeDy_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
# python PGEx_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
# python random_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
  ### explainers=subgraphx_tg ###
python TGNNEx_run.py datasets=${dataset} device_id=1 explainers=subgraphx_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}


  ### for sparsity = 0.2 ###
# val=0.1
# for factual in 0 1; 
# do
#     for existed in 0 1;
#     do
#     python QIEA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     python GA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     python random_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     python GreeDy_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     python PGEx_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     ### explainers=subgraphx_tg ###
#     python TGNNEx_run.py datasets=${dataset} device_id=1 explainers=subgraphx_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#     done
# done

  ### for sparsity: 0.1 to 0.5 ###
# for factual in 0 1; 
# do
#     for existed in 0 1;
#     do
#         for dataset in wikipedia reddit;
#         do
#             for model in tgat tgn;
#             do
#                 for val in $(seq 0.1 0.1 0.5) #sparsity
#                 do
#                   ### pg_explainer_tg ###            
#                 echo "Running command with value: $dataset $model $factual $existed $val"
#                 python QIEA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#                 python GA-TGX_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#                 python random_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#                 python GreeDy_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#                 python PGEx_run.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
                  ### explainers=subgraphx_tg ###
#                 python TGNNEx_run.py datasets=${dataset} device_id=1 explainers=subgraphx_tg models=${model} sparse_ratio=${val} factual=${factual} existed=${existed}
#                 done
#             done
#         done
#     done
# done

