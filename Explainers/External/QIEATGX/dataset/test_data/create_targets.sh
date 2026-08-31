datasets=("wikipedia" "reddit")
models=("tgat" "tgn")

dataset=wikipedia # wikipedia, reddit
model=tgat # tgat, tgn
val=0.1 # sparsity_ratio
factual=1 # type of explanation, factual: 1, counterfactual: 0
existed=1 # target event existance, exist: 1, nonexist: 0


for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for factual in 0 1;
        do
            for existed in 0 1;
            do
            python create_targets.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} factual=${factual} existed=${existed}
            python one_2hop.py datasets=${dataset} device_id=1 explainers=pg_explainer_tg models=${model} factual=${factual} existed=${existed}
            done
        done
    done
done