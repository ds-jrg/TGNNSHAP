import numpy as np
import time
import math
import random
def search_ex_graph(can_events,factual,genom):
    '''make expalation graph'''
    vector1 = np.array(can_events)
    if factual:
        vector2 = np.array(genom)
    else:
        vector2 = np.array([1 if i == 0 else 0 for i in genom])
    product_vector = vector1 * vector2
    use_events = product_vector[product_vector != 0]
    return use_events



def Quantum_individual_optimizationA(agent_iteration,model, link_data, can_events, H, teta,factual, existed,max_events,initial_output,flag,hops):
    '''Function for quantum individual optimization
    agents: Number of agents (individuals), population size
    model, link_data, can_events: Data required for individual evaluation
    H: Parameter for H-gate (0 < H < 1)
    iteration: Number of iterations
    teta: Quantum bit rotation angle
    '''
    model_time = []
    if 'ob' in flag:
        ''' Manage hop information with indices '''
        one_hop = hops[:,0]

        two_hop = hops[:, 1:] 
        # Obtain indices for 1-hop
        # Check for duplicates and remove them
        _, unique_indices = np.unique(one_hop, return_index=True)  # 重複なしのインデックスを取得
        unique_indices = np.sort(unique_indices)  # インデックスをソート

        # Create one_hop and two_hop without duplicates
        one_hop = one_hop[unique_indices]
        two_hop = two_hop[unique_indices]
        hop1_index = np.zeros(len(one_hop), dtype=int)
        for i in range(len(one_hop)):
            s = np.where(can_events == one_hop[i])[0]
            hop1_index[i] = int(s)

        # Obtain indices for 2-hop
        hop2_index = np.zeros(len(can_events)-len(one_hop), dtype=int)
        two_hop_unique = np.setdiff1d(can_events, one_hop) # ここちゃんと動いてる？？
        assert len(two_hop_unique) == len(can_events)-len(one_hop)
        for i in range(len(two_hop_unique)):
            _ =np.where(can_events == two_hop_unique[i])[0]
            hop2_index[i] = int(_)
        
        # Get connections of 2-hop to 1-hop
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


    genome_size = len(can_events)  
    up_down = int(factual==existed)
    agents = agent_iteration[0]
    iteration = agent_iteration[1]
    # Initialize quantum individuals (set all angles to π/4)
    Q = np.full((agents, genome_size), np.pi / 4)  # (agents, genome_size)
    p = np.zeros((agents, genome_size), dtype=int) 
    if up_down:
        B = np.zeros((agents, genome_size), dtype=int) 
    else:
        B = np.ones((agents, genome_size), dtype=int)
    fitness = np.zeros(agents)
    if up_down:
        best_agent_fitness = np.full(agents, -math.inf)  
        best_global_fitness = -math.inf  
    else:
        best_agent_fitness = np.full(agents, math.inf) 
        best_global_fitness = math.inf
    best_global_genome = np.zeros(genome_size, dtype=int)

    result_list = []
    loop_list = []
    ## Initial observation and fitness calculation (excluded from computation count)
    for i in range(agents):
        # p[i] = observe(Q[i])  # observation
        if flag == 'ob_hops':
            p[i] = observe_random_order_hops(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
        else:
            p[i] = observe_random_order(Q[i],factual,max_events)  # 観測
        if factual:
            assert sum(p[i]) <= max_events
        else:
            assert sum(p[i]) >= (genome_size-max_events)
        individual = Individual(p[i], model, link_data, can_events)
        model_time.append(individual.runtime)
        fitness[i] = individual.get_fitness()
        if up_down == 1:
            if fitness[i] > best_agent_fitness[i]:
                best_agent_fitness[i] = fitness[i]
                B[i] = p[i].copy()
            if fitness[i] > best_global_fitness:
                best_global_fitness = fitness[i]
                best_global_genome = p[i].copy()
        else:
            if fitness[i] < best_agent_fitness[i]:
                best_agent_fitness[i] = fitness[i]
                B[i] = p[i].copy()
            if fitness[i] < best_global_fitness:
                best_global_fitness = fitness[i]
                best_global_genome = p[i].copy()
    
    if factual:
        result_list.append([best_global_fitness,sum(best_global_genome),initial_output])    
    else:
        result_list.append([best_global_fitness,len(best_global_genome)-sum(best_global_genome),initial_output])

    half_iteration = int(iteration)
    teta = teta * np.ones(genome_size)
    if flag == 'half':
        half_iteration = int(iteration/2)
        teta2 = teta 
        teta = teta2 * 2
    
    
    if (flag == 'time') or (flag == 'time2'):
        sorted_indices = np.argsort(can_events)
        ch_para = 0.01
        min_value = 0
        max_value = ch_para
        if flag == 'time':
            scaled_events = np.linspace(min_value, max_value, len(can_events))
        else:
            scaled_events = np.linspace(max_value, min_value, len(can_events))
        scaled_events_in_order = [0] * len(can_events)
        for i, index in enumerate(sorted_indices):
            scaled_events_in_order[index] = scaled_events[i]
        
        teta2 = teta 

        scaled_events_in_order = np.array(scaled_events_in_order) * np.pi
        teta = teta2 + (scaled_events_in_order * np.pi)




    ab = 1
    # iteration = int(GENERATIONS / agents)
    for iter in range(iteration):
        if (iter == half_iteration):
            if (flag == 'half'):
                teta = teta2
            elif (flag == 'time') or (flag == 'time2'):
                teta = teta2

        # 各エージェントでやること
        for i in range(agents):
            # ii)観測
            if flag == 'ob_hops':
                p[i] = observe_random_order_hops(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)  
            elif flag == 'ob_half':
                a = np.random.choice([0, 1], p=[0.5, 0.5])
                if a == 0:
                    p[i] = observe_random_order(Q[i],factual,max_events)
                else:
                    p[i] = observe_random_order_hops(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
            elif flag == 'ob_hop_timeXtime':
                p[i] = observe_random_order_timeXtime(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
            elif flag == 'ob_hop_timeXtime_half':
                a = np.random.choice([0, 1], p=[0.5, 0.5])
                if a == 0:
                    p[i] = observe_random_order(Q[i],factual,max_events)
                else:
                    p[i] = observe_random_order_timeXtime(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
            elif flag == 'ob_hopXtime':
                a = np.random.choice([0, 1], p=[0.5, 0.5])
                if a == 0:
                    p[i] = observe_random_order_hops(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
                else:
                    p[i] = observe_random_order_timeXtime(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
            elif flag == 'ob_hop_all':
                roll = random.randint(1, 3)
                if roll == 1:
                    p[i] = observe_random_order(Q[i],factual,max_events)
                elif roll == 2:
                    p[i] = observe_random_order_hops(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
                else:
                    p[i] = observe_random_order_timeXtime(Q[i],factual,max_events,hop1_index,hop2_index,hop2_array)
            else:
                p[i] = observe_random_order(Q[i],factual,max_events)  # 観測
            individual = Individual(p[i], model, link_data, can_events)
            fitness[i] = individual.get_fitness()

            if up_down == 1: 
                df_q = []
                for j in range(len(Q[i])):
                    # original obsrve
                    if fitness[i] < best_agent_fitness[i]:
                        # P(i) に収束させるように更新
                        if B[i][j] == 1 and p[i][j] == 0:
                            delta_theta = teta[j]  
                        elif B[i][j] == 0 and p[i][j] == 1:
                            delta_theta = -teta[j]  
                        else:
                            delta_theta = 0
                    else:
                        delta_theta = 0
                    df_q.append(delta_theta)
                    # Q-bit updation
                    Q[i][j] += delta_theta
                # print(df_q)
            else: # 小さくする方向
                df_q = []
                for j in range(len(Q[i])):
                    if fitness[i] > best_agent_fitness[i]:
                        if B[i][j] == 1 and p[i][j] == 0:
                            delta_theta = teta[j]  
                        elif B[i][j] == 0 and p[i][j] == 1:
                            delta_theta = -teta[j] 
                        else:
                            delta_theta = 0
                    else:
                        delta_theta = 0
                    df_q.append(delta_theta)
                    Q[i][j] += delta_theta

            # H-gate Updation
            Q[i] = H_gate_min(Q[i], H)
            Q[i] = H_gate_max(Q[i], H)

            x_graph = search_ex_graph(can_events,factual,p[i])
            candiates_str = ','.join(map(str, x_graph))
            num_use_genom = sum(p[i])
            if factual==0:
                num_use_genom = len(best_global_genome) - num_use_genom
            result_list.append([fitness[i],num_use_genom,initial_output])
            loop_list.append([fitness[i],num_use_genom,candiates_str])

        for i in range(agents):
            if up_down == 1: 
                if fitness[i] > best_agent_fitness[i]:
                    best_agent_fitness[i] = fitness[i]
                    B[i] = p[i].copy()
                if fitness[i] > best_global_fitness:
                    best_global_fitness = fitness[i]
                    best_global_genome = p[i].copy()
            else: 
                if fitness[i] < best_agent_fitness[i]:
                    best_agent_fitness[i] = fitness[i]
                    B[i] = p[i].copy()
                if fitness[i] < best_global_fitness:
                    best_global_fitness = fitness[i]
                    best_global_genome = p[i].copy()
    result = []
    flag =0
    if factual == 0:
        if existed == 0:
            if best_global_fitness > 0.5:
                flag = 1
        else:
            if best_global_fitness < 0.5:
                flag = 1
    
    if factual:
        num_use_genom = sum(best_global_genome)
    else:
        num_use_genom = len(best_global_genome) - sum(best_global_genome)
    result=[flag,best_global_fitness,num_use_genom]
    return result,result_list,loop_list,model_time





def observe(Q):
    '''Observe quantum individuals to obtain binary genome'''
    prob = np.sin(Q) ** 2  
    random_values = np.random.rand(*Q.shape)  
    observed = random_values < prob  
    return observed.astype(int)  

def observe_random_order(Q,factual,max_events):
    '''A function to perform sampling in a random order 
    based on the probabilities of the Q vector'''
    n = len(Q)
    prob = np.sin(Q) ** 2 
    indices = np.arange(n)
    np.random.shuffle(indices) 
    observed = np.zeros(n, dtype=int)
    if factual == 0:
        observed = np.ones(n, dtype=int)
    
    count = 0
    for i in indices:
        random_value = np.random.rand()  
        observed[i] = 1 if random_value < prob[i] else 0  
        
        if factual: 
            if observed[i] == 1:
                count += 1
        else: 
            if observed[i] == 0:
                count += 1
        
        if count >= max_events:
                break
        
    return observed

def observe_random_order_hops(Q,factual,max_events,hop1,hop2,hop2_array):
    '''A function to perform sampling in a random order based on the probabilities of the Q vector'''  
    # hop1: Indices of can_events for 1-hop  
    # hop2: Indices of can_events for 2-hop
    
    n = len(Q)
    prob = np.sin(Q) ** 2  
    
    observed = np.zeros(n, dtype=int)
    if factual == 0:
        observed = np.ones(n, dtype=int)    

    
    count = 0
    hop1_ = hop1.copy()
    np.random.shuffle(hop1_)  
    for i in hop1_: 
        random_value = np.random.rand()  
        observed[i] = 1 if random_value < prob[i] else 0  
        
        if factual: 
            if observed[i] == 1:
                count += 1
        else: 
            if observed[i] == 0:
                count += 1
        
        if count >= max_events: 
            return observed
    
    # 2hopの観測
    hop2_ = hop2.copy()    
    np.random.shuffle(hop2_)  
    for i in hop2:         
        flag = 0 # observe or don't

        for j in range(len(hop2_array)): 
            if i in hop2_array[j]: 
                # Determine if a 1-hop has been observed
                for k in range(1,len(hop2_array[j])):
                    hop1_event = hop2_array[j][k]
                    if observed[hop1_event] == 1:
                        flag = 1
                        break
                break
        
        if flag == 1:
            random_value = np.random.rand()
            observed[i] = 1 if random_value < prob[i] else 0  
        else: #Forcefully observe
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


    return observed

def observe_random_order_timeXtime(Q,factual,max_events,hop1,hop2,hop2_array):
    '''A function to perform sampling in a random order based on the probabilities of the Q vector'''  
    # hop1: Indices of can_events for 1-hop  
    # hop2: Indices of can_events for 2-hop
    
    n = len(Q)
    prob = np.sin(Q) ** 2  
    
    observed = np.zeros(n, dtype=int)
    if factual == 0:
        observed = np.ones(n, dtype=int)   
    
    count = 0
    hop1_ = hop1.copy()
    hop1_sorted_indices = np.argsort(-hop1_)
    hop1_ = hop1_[hop1_sorted_indices]
    for i in hop1_: 
        random_value = np.random.rand() 
        observed[i] = 1 if random_value < prob[i] else 0 
        
        if factual: 
            if observed[i] == 1:
                count += 1
        else: 
            if observed[i] == 0:
                count += 1
        
        if count >= max_events: 
            return observed
    

    hop2_ = hop2.copy()    
    hop2_sorted_indices = np.argsort(-hop2)
    hop2_ = hop2_[hop2_sorted_indices]

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
            random_value = np.random.rand()
            observed[i] = 1 if random_value < prob[i] else 0  
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


    return observed


def H_gate_min(v, H):
    '''Minimum angle restriction by H-gate'''
    min_angle = np.arccos(np.sqrt(H))
    v = np.where(v > min_angle, min_angle, v)
    return v

def H_gate_max(v, H):
    '''Maximum angle restriction by H-gate'''
    max_angle = np.arccos(np.sqrt(1 - H))
    v = np.where(v < max_angle, max_angle, v)
    return v

class Individual:
    def __init__(self, genom, model, link_data, can_events):
        assert len(genom) == len(can_events)
        self.fitness = 0  
        self.genom = genom 
        self.set_fitness(model, link_data, can_events)
    
    def set_fitness(self, model, link_data, can_events):
        import numpy as np
        genom = self.genom
        vector1 = np.array(can_events)
        vector2 = np.array(genom)
        product_vector = vector1 * vector2
        use_events = product_vector[product_vector != 0]
        start_time = time.time()
        output1 = model.get_prob(*link_data, edge_idx_preserve_list=use_events)
        ori_ori_value = output1.cpu().detach().numpy()
        ori_cfevent_prob = ori_ori_value[0]
        self.runtime = time.time() - start_time
        self.fitness = ori_cfevent_prob.item()

    def get_fitness(self):
        return self.fitness
    
    def get_genom(self):
        return self.genom
    
    def mutate(self, model, link_data, can_events):
        import numpy as np
        tmp = self.genom.copy()
        i = np.random.randint(0, len(self.genom) - 1)
        tmp[i] = float(not self.genom[i])
        self.genom = tmp
        self.set_fitness(model, link_data, can_events)


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # オーバーフロー対策で最大値を引く
    return exp_x / np.sum(exp_x)




def greedy(model,link_data,can_events,up_down,factual):
    result_list = [[],[]]
    for can in can_events:
        if factual == 1:
            candiates_2hop = [can]
        else:
            candiates_2hop = [e_idx for e_idx in can_events if e_idx != can ]
        output2 = model.get_prob(*link_data, edge_idx_preserve_list=candiates_2hop)
        one_output = output2.cpu().detach().numpy()
        output1 = one_output[0]
        result_list[0].append(can)
        result_list[1].append(output1.item())
    if up_down:
        sorted_result = sorted(zip(result_list[0], result_list[1]), key=lambda x: x[1], reverse=True)
    else:
        sorted_result = sorted(zip(result_list[0], result_list[1]), key=lambda x: x[1], reverse=False)
    result_list[0], result_list[1] = zip(*sorted_result)
    return result_list
