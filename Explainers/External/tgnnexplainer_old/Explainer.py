from Explainers.utils import Explainer, ExplanationResult

from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler
from DyGLib.models.modules import TGNN

from Config.config import CONFIG

from .tgnnexplainer.tgnnexplainer.xgraph.method.subgraphx_tg import SubgraphXTG
from .tgnnexplainer.tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt
from .tgnnexplainer.tgnnexplainer.xgraph.evaluation.metrics_tg import EvaluatorMCTSTG

import numpy as np
import pandas as pd
import itertools

CONFIG = CONFIG()

class SubgraphXTExplainer(Explainer):
    def __init__(self, model:TGNN, neighbor_finder: NeighborSampler, data: Data):
        super().__init__(model, neighbor_finder, data)        

    def initialize(self):
        
        explainer = PGExplainerExt(
            self.model,
            self.neighbor_finder,
            model_name=CONFIG.model.model_name.lower(),
            explainer_name="pg_explainer_tg",  # fixed
            dataset_name=CONFIG.data.dataset_name,
            seed=123,
            all_events=self.data.dataset if self.data.dataset is not None else pd.DataFrame(),
            explanation_level="event",
            device=CONFIG.model.device,
            results_dir="Logs/TGNNExplainer/"+CONFIG.data.dataset_name,
            train_epochs=50,
            explainer_ckpt_dir=CONFIG.data.folder+"/checkpoints/pg_explainer_tg",
            reg_coefs=(0.5,0.1),
            batch_size=16,
            lr=1e-4,
            debug_mode=False,
        )
        explainer(event_idxs=[]) # Runs training if not already trained, otherwise loads the trained model from checkpoint
        
        pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(
            self.model,  # load a trained mlp model
            model_name=CONFIG.model.model_name.lower(),
            explainer_name="pg_explainer_tg",  # fixed
            dataset_name=CONFIG.data.dataset_name,
            ckpt_dir=CONFIG.data.folder+"/checkpoints/pg_explainer_tg",
            device=CONFIG.model.device,
            seed=123,
        )
        print("used pg_explainer_tg ckpt:", explainer_ckpt_path)
        
        self.explainer = SubgraphXTG(
                self.model,
                self.neighbor_finder,
                CONFIG.model.model_name.lower(),
                "subgraphx_tg",
                CONFIG.data.dataset_name,
                123,
                self.data.dataset if self.data.dataset is not None else pd.DataFrame(),
                "event",
                device=CONFIG.model.device,
                results_dir="Logs/TGNNExplainer/"+CONFIG.data.dataset_name,
                debug_mode=True,
                save_results=False,
                mcts_saved_dir="Logs/TGNNExplainer/"+CONFIG.data.dataset_name,
                load_results=False,
                rollout=CONFIG.tgnnExplainerConfig.num_rollouts,
                min_atoms=CONFIG.tgnnExplainerConfig.min_atoms,
                c_puct=5,
                pg_explainer_model=pg_explainer_model,
                pg_positive=True,
                num_layers=CONFIG.model.num_layers,
                num_neighbors=CONFIG.model.num_neighbors,
            )

        self.evaluator = EvaluatorMCTSTG(
            CONFIG.model.model_name,
            explainer_name="subgraphx_tg",
            dataset_name=CONFIG.data.dataset_name,
            explainer=self.explainer,
            results_dir=CONFIG.data.folder,
            seed=123,
            cpuct=5
        )

    def explain_instance(self, src, dst, timestamp, silent = False):
        mask = (self.data.src_node_ids == src) & (self.data.dst_node_ids == dst) & (self.data.node_interact_times == timestamp)
        edge_id = self.data.edge_ids[mask]

        if silent == True:
            self.explainer.debug_mode = False
            self.explainer.verbose = False
            self.evaluator.debug_mode = False

        explain_results = self.explainer(event_idxs=int(edge_id))

        coalitions = self.evaluator.evaluate(explain_results, edge_id)["coalition"]
        
        return coalitions
    
    def build_coalitions(self, explanation):

        coalitions = np.array(list(itertools.zip_longest(*explanation, fillvalue=0))).T
        base_events = np.unique(np.array(self.explainer.base_events)).reshape((1,-1))
        base_events = np.repeat(base_events, axis=0, repeats=coalitions.shape[0])
        coalitions = np.concat([base_events,coalitions], axis=1)
        
        return coalitions, None, None
        