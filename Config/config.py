from dataclasses import dataclass
import yaml
from typing import Optional
import os.path

from dataclasses import dataclass

@dataclass
class DataConfig:
    '''Configuration for the dataset'''
    dataset_name: str  
    '''Name of the dataset to be used. This mostly defines where models or preprocessed data is stored'''
    folder: str        
    '''Folder path where the dataset is stored'''
    is_directed: bool  
    '''Boolean indicating if the graph is directed'''
    edge_feat_file: str 
    '''File name with extention for edge features'''
    node_feat_file: str 
    '''File name with extention for node features'''
    index_file: str     
    '''File name with extention for graph strucuture'''
    feature_names_file: str 
    '''File name with extention for feature names. If blank, feature names are not used for plotting an explanation.'''
    node_names_file: str 
    '''File name with extention for node names. If blank, node names are not used for plotting an explanation.'''

@dataclass
class TrainConfig:
    '''Configuration for trainig the TGNN'''
    num_epochs: int        
    '''Number of epochs for training'''
    batch_size: int    
    '''Size of the batch for training and evaluation'''
    optimizer: str         
    '''Type of optimizer to use (e.g., 'SGD', 'Adam')'''
    weight_decay: float    
    '''Weight decay for regularization'''
    patience: int          
    '''Patience for early stopping'''
    learning_rate: float   
    '''Learning rate for the optimizer'''
    val_ratio: float       
    '''Ratio of the dataset to use for validation'''
    test_ratio: float      
    '''Ratio of the dataset to use for testing'''
    num_runs: int          
    '''Number of runs with different seeds'''
    test_interval_epochs: int 
    '''Interval of epochs to perform testing'''

@dataclass
class TempMEConfig:
    '''Configuration for TempME and the explainer model'''
    bs: int                
    '''Batch size for training'''
    n_epoch: int           
    '''Number of epochs for training'''
    out_dim: int           
    '''Output dimension for the model'''
    hid_dim: int           
    '''Hidden dimension for the model'''
    temp: float            
    '''Temperature parameter'''
    prior_p: float         
    '''Prior belief of sparsity'''
    lr: float              
    '''Learning rate'''
    drop_out: float        
    '''Dropout probability'''
    if_bern: bool          
    '''Boolean indicating if Bernoulli sampling is used'''
    weight_decay: float    
    '''Weight decay for regularization'''
    beta: float            
    '''Beta parameter'''
    device: str            
    '''Device to run the model on (e.g., 'cuda', 'cpu')'''

@dataclass
class ModelConfig:
    '''Configuration for the TGNN'''
    model_name: str        
    '''Name of the model to be used, possible: JODIE, DyRep, TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer'''
    device: str            
    '''Device to run the model on (e.g., 'cuda', 'cpu')'''
    num_neighbors: int     
    '''Number of neighbors to sample for each node'''
    sample_neighbor_strategy: str 
    '''Strategy for sampling historical neighbors'''
    time_scaling_factor: float 
    '''Hyperparameter controlling sampling preference with time interval'''
    num_walk_heads: int    
    '''Number of heads used for attention in walk encoder'''
    num_heads: int         
    '''Number of heads used in attention layer'''
    num_layers: int        
    '''Number of model layers'''
    num_reg_layers: int    
    '''Number of regularization layers'''
    hidden_reg_layers_dim: int 
    '''Dimension of hidden layers in regularization'''
    node_dim: int          
    '''Dimension of node features'''
    walk_length: int       
    '''Length of each random walk'''
    time_gap: int         
    '''Time gap for neighbors to compute node features'''
    time_feat_dim: int     
    '''Dimension of the time embedding'''
    position_feat_dim: int 
    '''Dimension of the position embedding'''
    patch_size: int        
    '''Patch size'''
    channel_embedding_dim: int 
    '''Dimension of each channel embedding'''
    max_input_sequence_length: int 
    '''Maximal length of the input sequence of each node'''
    dropout: float         
    '''Dropout rate'''
    task: str              
    '''Task to be performed (e.g., 'link prediction')'''
    trained_model_path: str 
    '''Path to the trained model. This can be set after training to inform the explainers where the model is located.'''
    omit_normalization: bool  
    '''Whether to omit cosine function and layer normalization in TGAT model (enables better extrapolation on regression tasks)'''

@dataclass
class TGNNExplainerConfig:
    '''Configuration for the TGNN explainer'''
    num_rollouts: int      
    '''Number of rollouts for MCTS'''
    min_atoms: int         
    '''Minimum number of atoms for explanation'''


@dataclass
class ShapleyConfig:
    '''Configuration for Shapley-based explainers'''
    batch_size: int
    '''Maximum number of model instances queried at once by the Shapley explainers'''
    max_num_samples: int
    '''Maximum number of coalition samples used for feature-level Monte Carlo explanations'''


@dataclass
class QIEAConfig:
    '''Configuration for the QIEA-TGX explainer.'''
    func: str = "ob_half"
    n_trials: int = 500
    pop_size: int = 20
    QIEA_agents: int = 1

class CONFIG:
    _instance = None
    model: ModelConfig
    data: DataConfig
    tempME: TempMEConfig
    tgnnExplainerConfig: TGNNExplainerConfig
    shapley: ShapleyConfig
    qiea: QIEAConfig
    train: TrainConfig


    def __new__(cls, config: Optional[str]= None):
        if cls._instance is None:
            assert config is not None, "Config file name (without extension) needs to be given!"
            filename = f"Config/{config}.yaml"
            assert os.path.isfile(filename), f"File {filename} not found!"
            cls._instance = super(CONFIG, cls).__new__(cls)
            with open(filename, 'r') as file:
                config = yaml.safe_load(file)
                cls._instance.model = ModelConfig(**config['ModelConfig']) # type: ignore
                cls._instance.data = DataConfig(**config['DataConfig']) # type: ignore
                cls._instance.tempME = TempMEConfig(**config['TempMEConfig']) # type: ignore
                cls._instance.tgnnExplainerConfig = TGNNExplainerConfig(**config['TGNNExplainerConfig']) # type: ignore
                cls._instance.shapley = ShapleyConfig(**config['Shapley']) # type: ignore
                cls._instance.qiea = QIEAConfig(**config.get('QIEA')) # type: ignore
                cls._instance.train = TrainConfig(**config['TrainConfig']) # type: ignore
        return cls._instance