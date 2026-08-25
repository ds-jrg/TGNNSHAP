from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from Explainers.External.TempME.Explainer import TempMEExplainer
from DyGLib.utils.DataLoader import get_link_prediction_data
    
node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
    get_link_prediction_data(val_ratio=0.1, test_ratio=0.1, node_dim=CONFIG.model.node_dim)
    
print("Preprocessing data for TempME...")
TempMEExplainer.preprocess_data_if_missing(train_data, subset_name="train")
TempMEExplainer.preprocess_data_if_missing(full_data, subset_name="test")