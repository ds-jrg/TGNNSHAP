import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
np.random.seed(1)

class Individual:
    '''各個体のクラス
        args: 個体の持つ遺伝子情報(np.array)'''
    def __init__(self, genom,model,link_data,can_events):
        assert len(genom) == len(can_events)
        self.fitness = 0  # 個体の適応度(set_fitness関数で設定)
        self.genom = genom # 個体の遺伝子情報
        self.set_fitness(model,link_data,can_events)
    

    def set_fitness(self,model,link_data, can_events):
        '''個体に対する目的関数の値をself.fitnessに代入'''
        genom = self.genom
        vector1 = np.array(can_events)
        vector2 = np.array(genom)
        # 要素ごとの積
        product_vector = vector1 * vector2
        # 0を除去したベクトル
        use_events = product_vector[product_vector != 0]
        output1 = model.get_prob( *link_data,  edge_idx_preserve_list=use_events)
        ori_ori_value = output1.cpu().detach().numpy()
        ori_cfevent_prob = ori_ori_value[0]
        self.fitness = ori_cfevent_prob.item()

    def get_fitness(self):
        '''self.fitnessを出力'''
        return self.fitness
    
    def get_genom(self):
        '''self.genomを出力'''
        return self.genom
    
    def mutate(self,model,link_data,can_events):
        '''遺伝子の突然変異'''
        tmp = self.genom.copy()
        i = np.random.randint(0, len(self.genom) - 1)
        tmp[i] = float(not self.genom[i])
        self.genom = tmp
        self.set_fitness(model,link_data,can_events)


def greedy(model,link_data,can_events,up_down,factual):
    '''貪欲法'''
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
    exp_x = np.exp(x - np.max(x))  # 数値安定性のために最大値を引きます
    return exp_x / exp_x.sum(axis=0, keepdims=True)



def select_roulette(generation,model,link_data,can_events,up_down):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    if up_down:
        weights = [ind.get_fitness() for ind in generation]
    else:
        weights = [(1/ind.get_fitness()) for ind in generation]
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected

def select_roulette_sm(generation,model,link_data,can_events,up_down):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    if up_down:
        weights = [ind.get_fitness() for ind in generation]
    else:
        weights = [ind.get_fitness() for ind in generation]
        weights = -weights
    
    weights = softmax(weights)
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected


def select_roulette_pg(generation,model,link_data,can_events,up_down,pg_values):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    # fitnessではなく，pg_dataを使って選択
    pg_val = []
    for ind in generation:
        genom = ind.get_genom()
        vector1 = np.array(pg_values)
        vector2 = np.array(genom)
        # 要素ごとの積
        product = (vector1 * vector2)
        if product.sum() == 0:
            product = 0
        elif vector2.sum() == 0:
            product = 0
        else:
            product = product.sum()/vector2.sum()
        pg_val.append(product)

    min_pg = min(pg_val)

    if min_pg < 0:
        pg_val = [(1 / (1 + np.exp(-val))) for val in pg_val]

    if up_down:
        weights = pg_val
    else:
        # pg_val = (pg_val - pg_val.min()) / (pg_val.max() - pg_val.min())
        weights = [(1/val) for val in pg_val]
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected

def greedy_roulette(generation,model,link_data,can_events,up_down,grdy_values):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    # fitnessではなく，pg_dataを使って選択
    pg_val = []
    for ind in generation:
        genom = ind.get_genom()
        vector1 = np.array(grdy_values)
        vector2 = np.array(genom)
        # 要素ごとの積
        product = (vector1 * vector2)
        if product.sum() == 0:
            product = 0
        elif vector2.sum() == 0:
            product = 0
        else:
            product = product.sum()/vector2.sum()
        pg_val.append(product)

    min_pg = min(pg_val)

    if min_pg <= 0:
        pg_val = [(1 / (1 + np.exp(-val))) for val in pg_val]

    if up_down:
        weights = pg_val
    else:
        # pg_val = (pg_val - pg_val.min()) / (pg_val.max() - pg_val.min())
        weights = [(1/val) for val in pg_val]
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected


def greedy_roulette_sm(generation,model,link_data,can_events,up_down,grdy_values):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    # fitnessではなく，pg_dataを使って選択
    pg_val = []
    assert all(val >= 0 for val in grdy_values)
    vector1 = np.array(grdy_values)
    # vector1 = softmax(vector1)
    for ind in generation:
        genom = ind.get_genom()
        vector2 = np.array(genom)
        # 要素ごとの積
        product = (vector1 * vector2)
        if product.sum() == 0:
            product = 0
        elif vector2.sum() == 0:
            product = 0
        else:
            product = product.sum()/vector2.sum()
        pg_val.append(product)

    # min_pg = min(pg_val)
    if up_down:
        weights = pg_val
        weights = softmax(weights)
    else:
        weights = -pg_val
        weights = softmax(weights)
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected

def greedy_sm_roulette(generation,model,link_data,can_events,up_down,grdy_values):
    '''選択の関数(ルーレット方式)'''
    # selected = []
    # fitnessではなく，pg_dataを使って選択
    pg_val = []
    assert all(val >= 0 for val in grdy_values)
    vector1 = np.array(grdy_values)
    # print(vector1)
    vector1 = softmax(vector1)
    for ind in generation:
        genom = ind.get_genom()
        vector2 = np.array(genom)
        # 要素ごとの積
        product = (vector1 * vector2)
        if product.sum() == 0:
            product = 0
        elif vector2.sum() == 0:
            product = 0
        else:
            product = product.sum()/vector2.sum()
        pg_val.append(product)

    # min_pg = min(pg_val)
    if up_down:
        weights = pg_val
    else:
        val_max = max(pg_val)
        weights =  [-pg_val[i] + val_max for i in range(len(pg_val))]
    
    norm_weights = weights / np.sum(weights)
    # norm_weights = [(1/ind.get_fitness()) / sum(weights) for ind in generation]
    # 一つ選ぶ
    selected = np.random.choice(generation, size=1, p=norm_weights)
    # exit()
    return selected



def select_tournament(generation,model,link_data,can_events,up_down):
    '''選択の関数(トーナメント方式)'''
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
    '''交叉の関数'''
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




def uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events):
    '''一様交叉'''
    children = []
    # if POPURATIONS % 2:
    #     selected.append(selected[0])
    best_ind = max(generation, key=Individual.get_fitness)
    CROSSOVER_PB = 0.5
    genom = selected[0].genom
    best_ind_genom = best_ind.genom
    # どちらかが全て0か全て1の場合の応急処置
    if (sum(genom) + sum(best_ind_genom) == 0) or ((sum(genom) + sum(best_ind_genom)) == (2*len(can_events))):
        if factual:
            array = [1] * 1 + [0] * (len(can_events) - 1)  # 配列をランダムに並べ替える
        else:
            array = [0] * 1 + [1] * (len(can_events) - 1)
        # 配列をランダムに並べ替える
        np.random.shuffle(array)
        return array
        # return Individual(array,model,link_data,can_events)
    
    # ここで時間がかかる可能性がある
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
    # new_child1 = Individual(child1,model,link_data,can_events)
    return child1    

def uniform_crossover_random(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events,select_pb = 1):
    '''一様交叉'''
    children = []
    # if POPURATIONS % 2:
    #     selected.append(selected[0])

    # best_ind = max(generation, key=Individual.get_fitness)
    if select_pb < 1:
        flag = np.random.choice([0,1], p=[1-select_pb, select_pb])
        if flag == 0:
            use_ind = np.random.choice(generation, size=1)
            use_ind = use_ind[0]
        else:
            use_ind = max(generation, key=Individual.get_fitness)   
    CROSSOVER_PB = 0.5
    genom = selected[0].genom
    best_ind_genom = use_ind.genom
    # どちらかが全て0か全て1の場合の応急処置
    if (sum(genom) + sum(best_ind_genom) == 0) or ((sum(genom) + sum(best_ind_genom)) == (2*len(can_events))):
        if factual:
            array = [1] * 1 + [0] * (len(can_events) - 1)  # 配列をランダムに並べ替える
        else:
            array = [0] * 1 + [1] * (len(can_events) - 1)
        # 配列をランダムに並べ替える
        np.random.shuffle(array)
        return array
        # return Individual(array,model,link_data,can_events)
    
    # ここで時間がかかる可能性がある
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
    # new_child1 = Individual(child1,model,link_data,can_events)
    return child1    



def uniform_crossover_pg(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,pg_data):
    '''一様交叉'''
    children = []
    # if POPURATIONS % 2:
    #     selected.append(selected[0])
    best_ind = max(generation, key=Individual.get_fitness)
    child1 = []
    

    # CROSSOVER_PB = 
    pg_data = [(1 / (1 + np.exp(-val))) for val in pg_data]
    i = 0
    for p1_gene, p2_gene in zip(selected[0].genom, best_ind.genom):
        if np.random.rand() < pg_data[i]:
            child1.append(p1_gene)
        else:
            child1.append(p2_gene)
        i += 1
    new_child1 = Individual(child1,model,link_data,can_events)
    return new_child1    



def cross_two_point_copy(child1, child2,model,link_data,can_events):
    '''二点交叉'''
    size = len(child1.genom)
    tmp1 = child1.genom.copy()
    tmp2 = child2.genom.copy()
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    new_child1 = Individual(tmp1,model,link_data,can_events)
    new_child2 = Individual(tmp2,model,link_data,can_events)
    return new_child1, new_child2

def elite_save(generation,child,POPURATIONS,up_down):
    '''エリート保存'''
    if up_down:
        generation = sorted(generation, key=Individual.get_fitness)
    else:
        generation = sorted(generation, key=Individual.get_fitness,reverse=True)
    generation.append(child)
    generation = generation[:POPURATIONS]
    return generation #ここで，保存が行われてるかを確認


    # elite = max(generation, key=Individual.get_fitness)
    # elite_index = generation.index(elite)
    # if elite_index == 0:
    #     generation.append(elite)


def mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events):
    '''突然変異'''
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
        # use_genom = sum(tmp)
        if factual:
            if sum(tmp) <= max_events:
                genom = tmp
                break
        else:
            if sum(tmp) >= (len(can_events) - max_events):
                genom = tmp
                break
    
    # individual = Individual(genom,model,link_data,can_events)
    return genom

def mutate2(generation,MUTATION_PB,model,link_data,can_events,factual,max_events,sparsity):
    '''全部を突然変異'''
    # max_eventsは事実では使う，反事実では使わないイベントの数
    cf_num = len(can_events) - max_events
    while True:
        if factual:
            # one_genom = np.random.randint(0, 2, len(can_events))
            one_genom = [1 if np.random.random() < sparsity else 0 for _ in range(len(can_events))]
            if sum(one_genom) <= max_events:
                if sum(one_genom) == 0:
                    continue
                break
        else:
            one_genom = np.random.randint(0, 2, len(can_events))
            one_genom = [1 if np.random.random() > sparsity else 0 for _ in range(len(can_events))]
            if sum(one_genom) >= cf_num:
                if sum(one_genom) == len(can_events):
                    continue
                break

    # individual = Individual(one_genom,model,link_data,can_events)
    return one_genom



def mutate_pg(generation,MUTATION_PB,model,link_data,can_events,pg_data):
    '''突然変異'''

    selected = np.random.choice(generation, size=1)
    genom = selected[0].get_genom()
    weights = [(1 / (1 + np.exp(-val))) for val in pg_data]
    norm_weights = weights / np.sum(weights)
    array = np.arange(0, len(can_events))
    genom_index = np.random.choice(array, size=1, p=norm_weights)
    genom[genom_index] = float(not genom[genom_index])
    individual = Individual(genom,model,link_data,can_events)

    return individual


def create_generation_grdy(POPURATIONS, GENOMS,model,link_data,can_events,sparsity,factual,max_events,seed,up_down):
    '''初期世代の作成
        return: 個体クラスのリスト'''
    np.random.seed(seed)
    generation = []
    cf_num = len(can_events) - max_events
    history = []
    can_list= greedy(model,link_data,can_events,up_down,factual)

    # 2hopの結果
    loop_num = POPURATIONS
    # max_eventsは事実では使う，反事実では使わないイベントの数
    if len(can_list[0]) <= POPURATIONS: # 20個より小さい場合は全て使う
        explain_events = can_list[0][:]
        loop_num = len(can_list[0])
    else: # 20個より大きい場合は，POPURATIONS個だけ使う
        if factual:
            explain_events = can_list[0][:POPURATIONS]
        else:
            explain_events = can_list[0][len(can_list[0])-POPURATIONS:]
    
    

    for i in range(loop_num):

        use_one_event = explain_events[i]
        if factual:
            one_genom = []
            for j in can_events:
                if j == use_one_event:
                    one_genom.append(1)
                else:
                    one_genom.append(0)
        else:
            one_genom = []
            for j in can_events:
                if i > len(can_events):
                    continue
                if j == use_one_event:
                    one_genom.append(0)
                else:
                    one_genom.append(1)

        history.append(one_genom)
        individual = Individual(one_genom,model,link_data,can_events)
        generation.append(individual)

    #generationがPOPURATIONSに満たない場合
    
    while True:
        while True:
            if factual:
                # one_genom = np.random.randint(0, 2, len(can_events))
                one_genom = [1 if np.random.random() < sparsity else 0 for _ in range(len(can_events))]
                if sum(one_genom) <= max_events:
                    break
            else:
                one_genom = np.random.randint(0, 2, len(can_events))
                one_genom = [1 if np.random.random() > sparsity else 0 for _ in range(len(can_events))]
                if sum(one_genom) >= cf_num:
                    break
        if len(generation) == POPURATIONS:
            break
        history.append(one_genom)
        individual = Individual(one_genom,model,link_data,can_events)
        generation.append(individual)
        

    assert len(generation) == POPURATIONS
    return generation,history

def create_generation(POPURATIONS, GENOMS,model,link_data,can_events,sparsity,factual,max_events,seed):
    '''初期世代の作成
        return: 個体クラスのリスト'''
    np.random.seed(seed)
    generation = []
    history = []
    # max_eventsは事実では使う，反事実では使わないイベントの数
    cf_num = len(can_events) - max_events
    # start_time = time.time()
    # for i in range(POPURATIONS):
    #     num_use_events = np.random.randint(0,max_events)
    #     if factual:
    #         candiate = np.random.choice(can_events, size=num_use_events, replace=False)
    #     else:
    #         cf_num = len(can_events)-num_use_events
    #         candiate = np.random.choice(can_events, size=cf_num, replace=False)

    #     one_genom = [1 if i in candiate else 0 for i in can_events]
    #     assert len(one_genom) == len(can_events)
    #     individual = Individual(one_genom,model,link_data,can_events)
    #     generation.append(individual)
    # p1 = time.time() - start_time
    # start_time = time.time()
    for i in range(POPURATIONS):
        while True:
            if factual:
                # one_genom = np.random.randint(0, 2, len(can_events))
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
    # p2 = time.time() - start_time
    # print('p1,p2:',p1,p2)
    
    return generation,history

def search_ex_graph(can_events,factual,genom):
    '''選択されたイベントのグラフを探索'''
    vector1 = np.array(can_events)
    if factual:
        vector2 = np.array(genom)
    else:
        vector2 = np.array([1 if i == 0 else 0 for i in genom])
    # 要素ごとの積
    product_vector = vector1 * vector2
    # 0を除去したベクトル
    use_events = product_vector[product_vector != 0]
    # output1 = model.get_prob( *link_data,  edge_idx_preserve_list=use_events)
    return use_events

def ga_solve(generation, GENERATIONS, POPURATIONS,CROSSOVER_PB, MUTATION_PB,model,link_data,can_events,up_down,sparsity,factual,existed,pg_data,pg_index,max_events,initial_output,seed,genom_history):
    '''遺伝的アルゴリズムのソルバー
        return: 最終世代の最高適応値の個体、最低適応値の個体'''
    np.random.seed(seed)
    best = []
    best_subgraph = []
    num_spar = int(sparsity*10)
    num_events = []
    # for i in range(1, int(sparsity*10+1)):
    #     num_events.append(max(1, int(round(len(can_events) * (i / 10)))))
    # num_events = max(1,int(round(len(can_events) * sparsity)))
    # --- Generation loop
    # print('Generation loop start.')
    result_list = []
    loop_list = []
    #初期世代の適応度を計算
    for j in range(POPURATIONS):
        generation[j].set_fitness(model,link_data,can_events)
    generation_history = generation

    # generationの中からfitnessが最も高いものを選択
    if factual == existed:
        best_individual = max(generation, key=lambda ind: ind.get_fitness())
    else:
        best_individual = min(generation, key=lambda ind: ind.get_fitness())
    
    result_list.append([best_individual.get_fitness(),sum(best_individual.get_genom()),initial_output,0,0])    
    # result_fitness = [ind.get_fitness() for ind in generation]
    # result_genomsize = [sum(ind.get_genom()) for ind in generation]
    

    for i in range(GENERATIONS):
        start_time = time.time()
        flag = 0
        # --- Step2. Selection (Roulette)
        # 使うイベントを選ぶ
        if np.random.rand() < MUTATION_PB: #変異か交叉か
            # if pg_index == 4:
            #     child = mutate2(generation,MUTATION_PB,model,link_data,can_events,factual,max_events,sparsity)
            # else:
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

        else: #交叉
            # if pg_index == 1:
            #     selected = select_roulette_pg(generation,model,link_data,can_events,up_down,pg_data)
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

    # 保存するものを初期化
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


        # 上昇の場合
        if up_down:
            # 適応度が最大のものか判断
            if generation_history[j].get_fitness() > num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom
        # 下降の場合
        else:
            # 適応度が最小のものを選択
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


    # history =  np.array(history)
    # for j in range(len(history)):
    #     if factual:
    #         num_use_genom = sum(history[j].genom)
    #     else:
    #         num_use_genom = len(history[j].genom) - sum(history[j].genom)

    #     for i in range(num_spar): # 5回分
    #         # 上昇の場合
    #         if up_down:
    #             c=1
    #             if num_use_genom <= num_events[i]: #使うイベント数が閾値以下の場合
    #                 # 適応度が最大のものか判断
    #                 if history[j].get_fitness() > num_best_fitness[i]:
    #                     num_best_fitness[i] = history[j].get_fitness()
    #                     num_best_genoms[i] = num_use_genom
    #         # 下降の場合
    #         else:
    #             c=1
    #             if num_use_genom <= num_events[i]:
    #                 # 適応度が最小のものを選択
    #                 if history[j].get_fitness() < num_best_fitness[i]:
    #                     num_best_fitness[i] = history[j].get_fitness()
    #                     num_best_genoms[i] = num_use_genom

    # result = []
    # flag =0
    # for i in range(num_spar):
    #     if factual == 0:
    #         if existed == 0:
    #             if num_best_fitness[i] > 0.5:
    #                 flag = 1
    #         else:
    #             if num_best_fitness[i] < 0.5:
    #                 flag = 1
    #     result.append([flag,num_best_fitness[i],num_best_genoms[i]])

    return result,result_list,loop_list
    # if up_down:
    #     best_prob = max(best)
    # else:
    #     best_prob = min(best)

def ga_solve_grdy(generation, GENERATIONS, POPURATIONS,CROSSOVER_PB, MUTATION_PB,model,link_data,can_events,up_down,sparsity,factual,existed,grdy_data,grdy_index,max_events,initial_output,seed,genom_history):
    '''遺伝的アルゴリズムのソルバー
        return: 最終世代の最高適応値の個体、最低適応値の個体
        grdy_index: 1:greedyを初期世代に, 2:greedyを選択に（割合で計算）, 3:greedyを選択に（softmaxで計算）, 4:GA
    '''
    
    
    np.random.seed(seed)
    best = []
    best_subgraph = []
    num_spar = int(sparsity*10)
    num_events = []
    # for i in range(1, int(sparsity*10+1)):
    #     num_events.append(max(1, int(round(len(can_events) * (i / 10)))))
    # num_events = max(1,int(round(len(can_events) * sparsity)))
    # --- Generation loop
    # print('Generation loop start.')
    result_list = []
    loop_list = []
    #初期世代の適応度を計算
    if grdy_index in [2,3,4]: 
        can_list= greedy(model,link_data,can_events,up_down,factual)
        can_events = can_list[0]
        grdy_data = can_list[1]

    for j in range(POPURATIONS):
        generation[j].set_fitness(model,link_data,can_events)
    generation_history = generation
    


    # generationの中からfitnessが最も高いものを選択
    if factual == existed:
        best_individual = max(generation, key=lambda ind: ind.get_fitness())
    else:
        best_individual = min(generation, key=lambda ind: ind.get_fitness())
    
    # 最初のやつを保存
    result_list.append([best_individual.get_fitness(),sum(best_individual.get_genom()),initial_output,0,0])    
    # result_fitness = [ind.get_fitness() for ind in generation]
    # result_genomsize = [sum(ind.get_genom()) for ind in generation]
    

    for i in range(GENERATIONS):
        start_time = time.time()
        flag = 0
        # --- Step2. Selection (Roulette)
        # 使うイベントを選ぶ
        if np.random.rand() < MUTATION_PB: #変異か交叉か
            if grdy_index >= 5:
                AssertionError('Not implemented')
                # child = mutate2(generation,MUTATION_PB,model,link_data,can_events,factual,max_events,sparsity)
            else:
                counter = 0
                flag = 1
                while True:
                    # genomが帰ってくる
                    child = mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
                    if child in genom_history:
                        counter += 1
                        if counter > 50:
                            break
                    else:
                        break
                # child = mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
                # child = mutate2(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
            # generation = elite_save(generation,child,POPURATIONS,up_down)
            # genom_history.append(child)
        else: #交叉
            flag = 2
            counter = 0
            while True:
                counter += 1
                if grdy_index == 2:
                    selected = greedy_roulette(generation,model,link_data,can_events,up_down,grdy_data)
                elif grdy_index == 3:
                    selected = greedy_roulette_sm(generation,model,link_data,can_events,up_down,grdy_data)
                elif grdy_index == 4:
                    selected = greedy_sm_roulette(generation,model,link_data,can_events,up_down,grdy_data)
                else:
                    selected = select_roulette(generation,model,link_data,can_events,up_down)
                if grdy_index > 4:
                    AssertionError('Not implemented')
                    # child = uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,pg_data)
                else:
                    # genomが帰ってくる
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

    # 保存するものを初期化
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


        # 上昇の場合
        if up_down:
            # 適応度が最大のものか判断
            if generation_history[j].get_fitness() > num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom
        # 下降の場合
        else:
            # 適応度が最小のものを選択
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

# この関数はその時々で書き換える
def ga_solve_all(generation, GENERATIONS, POPURATIONS,CROSSOVER_PB, MUTATION_PB,model,link_data,can_events,up_down,sparsity,factual,existed,pg_data,pg_index,max_events,initial_output,seed,genom_history,cr_index,best_ind_pb):
    '''遺伝的アルゴリズムのソルバー
        return: 最終世代の最高適応値の個体、最低適応値の個体'''
    np.random.seed(seed)
    best = []
    best_subgraph = []
    num_spar = int(sparsity*10)
    num_events = []
    # for i in range(1, int(sparsity*10+1)):
    #     num_events.append(max(1, int(round(len(can_events) * (i / 10)))))
    # num_events = max(1,int(round(len(can_events) * sparsity)))
    # --- Generation loop
    # print('Generation loop start.')
    result_list = []
    loop_list = []
    #初期世代の適応度を計算
    for j in range(POPURATIONS):
        generation[j].set_fitness(model,link_data,can_events)
    generation_history = generation

    # generationの中からfitnessが最も高いものを選択
    if factual == existed:
        best_individual = max(generation, key=lambda ind: ind.get_fitness())
    else:
        best_individual = min(generation, key=lambda ind: ind.get_fitness())
    
    result_list.append([best_individual.get_fitness(),sum(best_individual.get_genom()),initial_output,0,0])    
    # result_fitness = [ind.get_fitness() for ind in generation]
    # result_genomsize = [sum(ind.get_genom()) for ind in generation]
    

    for i in range(GENERATIONS):
        start_time = time.time()
        flag = 0
        # --- Step2. Selection (Roulette)
        # 使うイベントを選ぶ
        if np.random.rand() < MUTATION_PB: #変異か交叉か
            # if pg_index == 4:
            #     child = mutate2(generation,MUTATION_PB,model,link_data,can_events,factual,max_events,sparsity)
            # else:
            counter = 0
            flag = 1
            # print("Mutation")
            while True:
                child = mutate(generation,MUTATION_PB,model,link_data,can_events,factual,max_events)
                if child in genom_history:
                    counter += 1
                    if counter > 20:
                        break
                else:
                    break

        else: #交叉
            # if pg_index == 1:
            #     selected = select_roulette_pg(generation,model,link_data,can_events,up_down,pg_data)
            flag = 2
            counter = 0
            # print("Crossover")
            while True:
                counter += 1
                if pg_index == 1:
                    selected = select_roulette_sm(generation,model,link_data,can_events,up_down)
                else:
                    selected = select_roulette(generation,model,link_data,can_events,up_down)
                if cr_index == 1:
                    child = uniform_crossover_random(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events,best_ind_pb)
                else:
                    child = uniform_crossover(selected,CROSSOVER_PB,POPURATIONS,model,link_data,can_events,generation,factual,max_events)
                if child in genom_history:
                    if counter > 20:
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

    # 保存するものを初期化
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


        # 上昇の場合
        if up_down:
            # 適応度が最大のものか判断
            if generation_history[j].get_fitness() > num_best_fitness:
                num_best_fitness = generation_history[j].get_fitness()
                num_best_genoms = num_use_genom
        # 下降の場合
        else:
            # 適応度が最小のものを選択
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
    # if up_down:
    #     best_prob = max(best)
    # else:
    #     best_prob = min(best)
