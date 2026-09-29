from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from Explainers.External.TempME.Explainer import TempMEExplainer
from DyGLib.utils.DataLoader import get_link_prediction_data
from DyGLib.utils.utils import get_neighbor_sampler
    
node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
    get_link_prediction_data(val_ratio=0.1, test_ratio=0.1, node_dim=CONFIG.model.node_dim)
    

train_neighbor_sampler = get_neighbor_sampler(data=train_data, edge_features=edge_raw_features, sample_neighbor_strategy=CONFIG.model.sample_neighbor_strategy,
                                                    time_scaling_factor=CONFIG.model.time_scaling_factor, seed=1)
    
full_neighbor_sampler = get_neighbor_sampler(data=full_data, edge_features=edge_raw_features, sample_neighbor_strategy=CONFIG.model.sample_neighbor_strategy,
                                                    time_scaling_factor=CONFIG.model.time_scaling_factor, seed=1)    
print("Preprocessing data for TempME...")
TempMEExplainer.preprocess_data_if_missing(train_data, train_neighbor_sampler, subset_name="train")
TempMEExplainer.preprocess_data_if_missing(full_data, full_neighbor_sampler, subset_name="test")