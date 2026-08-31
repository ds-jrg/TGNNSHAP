import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
np.random.seed(1)

class Individual:

    def __init__(self, genom,model,link_data,can_events):
        assert len(genom) == len(can_events)
        self.fitness = 0 
        self.genom = genom 
        self.set_fitness(model,link_data,can_events)
    

    def set_fitness(self,model,link_data, can_events):
        genom = self.genom
        vector1 = np.array(can_events)
        vector2 = np.array(genom)
        product_vector = vector1 * vector2
        use_events = product_vector[product_vector != 0]
        output1 = model.get_prob( *link_data,  edge_idx_preserve_list=use_events)
        ori_ori_value = output1.cpu().detach().numpy()
        ori_cfevent_prob = ori_ori_value[0]
        self.fitness = ori_cfevent_prob.item()

    def get_fitness(self):
        return self.fitness
    
    def get_genom(self):
        return self.genom
    
    def mutate(self,model,link_data,can_events):
        tmp = self.genom.copy()
        i = np.random.randint(0, len(self.genom) - 1)
        tmp[i] = float(not self.genom[i])
        self.genom = tmp
        self.set_fitness(model,link_data,can_events)


def greedy(model,link_data,can_events,up_down,factual):
    result_list = [[],[]]
    for can in can_events:
        if factual == 1:
            candiates_2hop = [can]
        else:
            # candiatesの中から、一つを取り除いたcoalitionを作成
            candiates_2hop = [e_idx for e_idx in can_events if e_idx != can ]
        output2 = model.get_prob(*link_data, edge_idx_preserve_list=candiates_2hop)
        # output1 = model.get_prob( src_idx_l, target_idx_l, cut_time_l,  edge_idx_preserve_list=events0)
        one_output = output2.cpu().detach().numpy()
        output1 = one_output[0]
        result_list[0].append(can)
        result_list[1].append(output1.item())
    # result_listをresult_list[1]の値に基づいてソートする
    # ここでソートが正しいかを確認する
    if up_down:
        sorted_result = sorted(zip(result_list[0], result_list[1]), key=lambda x: x[1], reverse=True)
    else:
        sorted_result = sorted(zip(result_list[0], result_list[1]), key=lambda x: x[1], reverse=False)
    result_list[0], result_list[1] = zip(*sorted_result)
    return result_list

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / exp_x.sum(axis=0, keepdims=True)



def select_roulette(generation,model,link_data,can_events,up_down):
    if up_down:
        weights = [ind.get_fitness() for ind in generation]
    else:
        weights = [(1/ind.get_fitness()) for ind in generation]
    
    norm_weights = weights / np.sum(weights)
    selected = np.random.choice(generation, size=1, p=norm_weights)
    return selected

def select_roulette_sm(generation,model,link_data,can_events,up_down):
    if up_down:
        weights = [ind.get_fitness() for ind in generation]
    else:
        weights = [ind.get_fitness() for ind in generation]
        weights = -weights
    
    weights = softmax(weights)
    
    norm_weights = weights / np.sum(weights)

    selected = np.random.choice(generation, size=1, p=norm_weights)
    return selected




def select_tournament(generation,model,link_data,can_events,up_down):
    selected = []
    for i in range(len(generation)):
        tournament = np.random.choice(generation, 3, replace=False)
        if up_down:
            max_genom = max(tournament, key=Individual.get_fitness).genom.copy()
        else:
            max_genom = min(tournament, key=Individual.get_fitness).genom.copy()
        selected.append(Individual(max_genom,model,link_data,can_events))
    return selected


def crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events):
    children = []
    if POPURATIONS % 2:
        selected.append(selected[0])
    for child1, child2 in zip(selected[::2], selected[1::2]):
        if np.random.rand() < CROSSOVER_PB:
            if len(child1.genom) >= 3:            
                child1, child2 = cross_two_point_copy(child1, child2,model,link_data,can_events)
        children.append(child1)
        children.append(child2)
    children = children[:POPURATIONS]
    return children


def get_hops(hops,can_events):
    one_hop = hops[:,0]
    two_hop = hops[:, 1:] 
    # Get the index of 1-hop
    # Check for duplicates and remove them
    _, unique_indices = np.unique(one_hop, return_index=True)  
    unique_indices = np.sort(unique_indices)  

    # Create one_hop and two_hop by removing duplicates
    one_hop = one_hop[unique_indices]
    two_hop = two_hop[unique_indices]
    hop1_index = np.zeros(len(one_hop), dtype=int)
    for i in range(len(one_hop)):
        s = np.where(can_events == one_hop[i])[0]
        hop1_index[i] = int(s)

    # Get the index of 2-hop
    hop2_index = np.zeros(len(can_events)-len(one_hop), dtype=int)
    two_hop_unique = np.setdiff1d(can_events, one_hop) 
    assert len(two_hop_unique) == len(can_events)-len(one_hop)
    for i in range(len(two_hop_unique)):
        _ =np.where(can_events == two_hop_unique[i])[0]
        hop2_index[i] = int(_)
    
    # Get the connection between 2-hop and 1-hop
    hop2_array = []
    for i in hop2_index:
        _list = []
        candidate = can_events[i]
        _list.append(i)
        for j in range(len(two_hop)):
            row_2hops = two_hop[j]
            if candidate in row_2hops:
                _list.append(hop1_index[j])
        hop2_array.append(_list)
    
    return hop1_index,hop2_index,hop2_array


def uniform_crossoverA(selected,CROSSOVER_PB,can_events,generation,factual,max_events,hop1,hop2,hop2_array):
    children = []
    best_ind = max(generation, key=Individual.get_fitness)
    CROSSOVER_PB = 0.5
    genom = selected[0].genom
    best_ind_genom = best_ind.genom
    # Temporary fix when either is all 0 or all 1
    if (sum(genom) + sum(best_ind_genom) == 0) or ((sum(genom) + sum(best_ind_genom)) == (2*len(can_events))):
        if factual:
            array = [1] * 1 + [0] * (len(can_events) - 1)  
        else:
            array = [0] * 1 + [1] * (len(can_events) - 1)
        np.random.shuffle(array)
        return array
    
    '''start'''
    observed = np.zeros(len(genom), dtype=int)
    if factual == 0:
        observed = np.ones(len(genom), dtype=int) 
    count = 0
    hop1_ = hop1.copy()
    np.random.shuffle(hop1_) 
    for i in hop1_: 
        if np.random.rand() < CROSSOVER_PB:
            observed[i] = genom[i]
        else:
            observed[i] = best_ind_genom[i]
        if factual: 
            if observed[i] == 1:
                count += 1
        else: 
            if observed[i] == 0:
                count += 1
        if count >= max_events: 
            return observed.tolist()
    
    # 2hop
    hop2_ = hop2.copy()    
    np.random.shuffle(hop2_)  
    for i in hop2: 
        
        flag = 0 
        for j in range(len(hop2_array)): 
            if i in hop2_array[j]: 
                for k in range(1,len(hop2_array[j])):
                    hop1_event = hop2_array[j][k]
                    if observed[hop1_event] == 1:
                        flag = 1
                        break
                break
        
        if flag == 1:
            if np.random.rand() < CROSSOVER_PB:
                observed[i] = genom[i]
            else:
                observed[i] = best_ind_genom[i]
        else: 
            if factual:
                observed[i] = 0
            else:
                observed[i] = 1
        
        if factual: 
            if observed[i] == 1:
                count += 1
        else: 
            if observed[i] == 0:
                count += 1
        
        if count >= max_events: 
            break
    return observed.tolist()


# vanila-GA
def uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events):
    children = []
    best_ind = max(generation, key=Individual.get_fitness)
    CROSSOVER_PB = 0.5
    genom = selected[0].genom
    best_ind_genom = best_ind.genom
    if (sum(genom) + sum(best_ind_genom) == 0) or ((sum(genom) + sum(best_ind_genom)) == (2*len(can_events))):
        if factual:
            array = [1] * 1 + [0] * (len(can_events) - 1) 
        else:
            array = [0] * 1 + [1] * (len(can_events) - 1)
        np.random.shuffle(array)
        return array
    
    while True:
        child1 = []
        for p1_gene, p2_gene in zip(genom, best_ind_genom):
            if np.random.rand() < CROSSOVER_PB:
                child1.append(p1_gene)
            else:
                child1.append(p2_gene)
        assert len(child1) == len(genom)

        if factual:
            if sum(child1) == 0:
                continue
            if sum(child1) <= max_events:
                break
        else:
            if sum(child1) == len(can_events):
                continue
            if sum(child1) >= (len(can_events) - max_events):
                break
    return child1    







def elite_save(generation,child,POPURATIONS,up_down):
    if up_down:
        generation = sorted(generation, key=Individual.get_fitness)
    else:
        generation = sorted(generation, key=Individual.get_fitness,reverse=True)
    generation.append(child)
    generation = generation[:POPURATIONS]
    return generation



def mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events):
    a = 1
    while True:
        selected = np.random.choice(generation, size=1)
        genom = selected[0].get_genom()
        if factual:
            if sum(genom) <= max_events:
                break
        else:
            if sum(genom) >= (len(can_events) - max_events):
                break

    s = 1            
    while True:
        tmp = genom.copy()
        i = np.random.randint(0, len(genom))
        tmp[i] = int(not genom[i])
        if factual:
            if sum(tmp) <= max_events:
                genom = tmp
                break
        else:
            if sum(tmp) >= (len(can_events) - max_events):
                genom = tmp
                break
    
    return genom

def create_generation(POPURATIONS, GENOMS,model,link_data,can_events,sparsity,factual,max_events,seed):

    np.random.seed(seed)
    generation = []
    history = []
    cf_num = len(can_events) - max_events
    for i in range(POPURATIONS):
        while True:
            if factual:
                one_genom = [1 if np.random.random() < sparsity else 0 for _ in range(len(can_events))]
                if sum(one_genom) <= max_events:
                    break
            else:
                one_genom = np.random.randint(0, 2, len(can_events))
                one_genom = [1 if np.random.random() > sparsity else 0 for _ in range(len(can_events))]
                if sum(one_genom) >= cf_num:
                    break

        history.append(one_genom)
        individual = Individual(one_genom,model,link_data,can_events)
        generation.append(individual)
    
    return generation,history

def search_ex_graph(can_events,factual,genom):
    vector1 = np.array(can_events)
    if factual:
        vector2 = np.array(genom)
    else:
        vector2 = np.array([1 if i == 0 else 0 for i in genom])
    product_vector = vector1 * vector2
    use_events = product_vector[product_vector != 0]
    return use_events


def ga_solveA(generation, GENERATIONS, POPURATIONS,CROSSOVER_PB, MUTATION_PB,model,link_data,can_events,up_down,sparsity,factual,existed,pg_data,pg_index,max_events,initial_output,seed,genom_history,hops):
    np.random.seed(seed)

    hop1_index,hop2_index,hop2_array = get_hops(hops,can_events)


    best = []
    best_subgraph = []
    num_spar = int(sparsity*10)
    num_events = []
    result_list = []
    loop_list = []
    for j in range(POPURATIONS):
        generation[j].set_fitness(model,link_data,can_events)
    generation_history = generation

    if factual == existed:
        best_individual = max(generation, key=lambda ind: ind.get_fitness())
    else:
        best_individual = min(generation, key=lambda ind: ind.get_fitness())
    
    result_list.append([best_individual.get_fitness(),sum(best_individual.get_genom()),initial_output,0,0])    

    for i in range(GENERATIONS):
        start_time = time.time()
        flag = 0
        if np.random.rand() < MUTATION_PB: 
            counter = 0
            flag = 1
            while True:
                child = mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
                if child in genom_history:
                    counter += 1
                    if counter > 50:
                        break
                else:
                    break

        else:
            flag = 2
            selected = select_roulette(generation,model,link_data,can_events,up_down)
            # child = uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events)
            child = uniform_crossoverA(selected,CROSSOVER_PB,can_events,generation,factual,max_events,hop1_index,hop2_index,hop2_array)

        
        genom_history.append(child)
        individual = Individual(child,model,link_data,can_events)
        generation_history.append(individual)
        generation = elite_save(generation,individual,POPURATIONS,up_down)   

        runtime = time.time() - start_time
        num_use_events = sum(individual.get_genom())
        output1 = individual.get_fitness()

        x_graph = search_ex_graph(can_events,factual,child)
        candiates_str = ','.join(map(str, x_graph))
        
        result_list.append([output1,num_use_events,initial_output,runtime,flag])
        loop_list.append([output1,len(x_graph),candiates_str])

    num_best_genoms = []
    if up_down:
        num_best_fitness = 0
    else:
        num_best_fitness = 1

    for j in range(len(generation_history)):
        if factual:
            num_use_genom = sum(generation_history[j].genom)
        else:
            num_use_genom = len(generation_history[j].genom) - sum(generation_history[j].genom)


        if up_down:
            if generation_history[j].get_fitness() > num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom
        else:
            if generation_history[j].get_fitness() < num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom

    result = []
    flag =0
    if factual == 0:
        if existed == 0:
            if num_best_fitness > 0.5:
                flag = 1
        else:
            if num_best_fitness < 0.5:
                flag = 1
    result=[flag,num_best_fitness,num_best_genoms]


    return result,result_list,loop_list


def ga_solve(generation, GENERATIONS, POPURATIONS,CROSSOVER_PB, MUTATION_PB,model,link_data,can_events,up_down,sparsity,factual,existed,pg_data,pg_index,max_events,initial_output,seed,genom_history):

    np.random.seed(seed)
    best = []
    best_subgraph = []
    num_spar = int(sparsity*10)
    num_events = []
    result_list = []
    loop_list = []
    for j in range(POPURATIONS):
        generation[j].set_fitness(model,link_data,can_events)
    generation_history = generation

    if factual == existed:
        best_individual = max(generation, key=lambda ind: ind.get_fitness())
    else:
        best_individual = min(generation, key=lambda ind: ind.get_fitness())
    
    result_list.append([best_individual.get_fitness(),sum(best_individual.get_genom()),initial_output,0,0])    
    

    for i in range(GENERATIONS):
        start_time = time.time()
        flag = 0
        if np.random.rand() < MUTATION_PB: 
            counter = 0
            flag = 1
            while True:
                child = mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
                if child in genom_history:
                    counter += 1
                    if counter > 50:
                        break
                else:
                    break

        else: 
            flag = 2
            counter = 0
            while True:
                counter += 1
                if pg_index == 1:
                    selected = select_roulette_sm(generation,model,link_data,can_events,up_down)
                else:
                    selected = select_roulette(generation,model,link_data,can_events,up_down)
                child = uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events)
                if child in genom_history:
                    if counter > 50:
                        break
                else:
                    break
        
        genom_history.append(child)
        individual = Individual(child,model,link_data,can_events)
        generation_history.append(individual)
        generation = elite_save(generation,individual,POPURATIONS,up_down)   

        runtime = time.time() - start_time
        num_use_events = sum(individual.get_genom())
        output1 = individual.get_fitness()

        x_graph = search_ex_graph(can_events,factual,child)
        candiates_str = ','.join(map(str, x_graph))
        
        result_list.append([output1,num_use_events,initial_output,runtime,flag])
        loop_list.append([output1,len(x_graph),candiates_str])

    num_best_genoms = []
    if up_down:
        num_best_fitness = 0
    else:
        num_best_fitness = 1

    for j in range(len(generation_history)):
        if factual:
            num_use_genom = sum(generation_history[j].genom)
        else:
            num_use_genom = len(generation_history[j].genom) - sum(generation_history[j].genom)


        if up_down:
            if generation_history[j].get_fitness() > num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom
        else:
            if generation_history[j].get_fitness() < num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom

    result = []
    flag =0
    if factual == 0:
        if existed == 0:
            if num_best_fitness > 0.5:
                flag = 1
        else:
            if num_best_fitness < 0.5:
                flag = 1
    result=[flag,num_best_fitness,num_best_genoms]

    return result,result_list,loop_list


