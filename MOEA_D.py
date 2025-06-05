import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Целевые функции
def f1(point):
    return 2 * point[1] * point[3] + point[2] * (point[0] - 2 * point[3])

def f2(point):
    P = 600
    L = 200
    E = 2 * 10**4
    I = (
        point[2] * (point[0] - 2 * point[3])**3 +
        2 * point[1] * point[3] * (4 * point[3]**2 + 3 * point[0] * (point[0] - 2 * point[3]))
    ) / 12
    return (P * L**3) / (48 * E * I)

# Проверка прочности 
def prov_sigma(p):
    Mz = 25000
    sigma = 16
    Wz = ((p[0] - 2 * p[3]) * p[2]**3 + 2 * p[3] * p[1]**3) / (6 * p[1])
    return Mz / Wz <= sigma

def sbx_crossover(p1, p2, bounds, eta=20):
    rand = np.random.rand(len(p1))
    beta = np.where(rand <= 0.5, (2 * rand)**(1/(eta + 1)), (1/(2*(1 - rand)))**(1/(eta + 1)))
    child1 = 0.5*((1 + beta)*p1 + (1 - beta)*p2)
    child2 = 0.5*((1 - beta)*p1 + (1 + beta)*p2)
    return (
        np.clip(child1, [b[0] for b in bounds], [b[1] for b in bounds]),
        np.clip(child2, [b[0] for b in bounds], [b[1] for b in bounds])
    )

def polynomial_mutation(child, bounds, mutation_rate=0.1, eta=100):
    if np.random.rand() < mutation_rate:
        normalized = [(child[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(child))]
        delta = [
            (1 - n)**(1/(eta + 1)) if np.random.rand() < 0.5 else n**(1/(eta + 1))
            for n in normalized
        ]
        mutated = [
            bounds[i][0] + (normalized[i] + (np.random.uniform(-1,1)*delta[i]))*(bounds[i][1]-bounds[i][0])
            for i in range(len(child))
        ]
        return np.clip(mutated, [b[0] for b in bounds], [b[1] for b in bounds])
    return child

def generate_weights(pop_size):
    return np.array([[i, 1-i] for i in np.linspace(0, 1, pop_size)])


def tchebycheff(f, weight, z):
    return max(np.abs(f - z) * weight)

# MAIN
population_size = 1500
generations = 2000
neighborhood_size = 20
bounds = np.array([[10, 80], [10, 50], [1, 5], [1, 5]]) 

population = []
while len(population) < population_size:
    candidate = np.array([np.random.uniform(low=bound[0], high=bound[1]) for bound in bounds])
    if prov_sigma(candidate):
        population.append(candidate)
population = np.array(population)

weights = generate_weights(population_size)
dist_matrix = cdist(weights, weights, 'euclidean')
neighbors = np.argsort(dist_matrix, axis=1)[:, :neighborhood_size]

z = np.array([min(f1(p) for p in population)] + 
             [min(f2(p) for p in population)])

current_solutions = population.copy()
current_f = np.array([[f1(p), f2(p)] for p in current_solutions])

for gen in range(generations):
    print(f'Generation: {gen+1}/{generations}')
    
    for i in range(population_size):
        parents_idx = np.random.choice(neighbors[i], 2, replace=False)
        parents = current_solutions[parents_idx]
        
        child1, child2 = sbx_crossover(parents[0], parents[1], bounds) 
        child = polynomial_mutation(child1, bounds)  
        
        if not prov_sigma(child):
            continue
            
        new_f = np.array([f1(child), f2(child)])
        z = np.minimum(z, new_f)
        
        for j in neighbors[i]:
            current_fj = current_f[j]
            tch_child = tchebycheff(new_f, weights[j], z)
            tch_current = tchebycheff(current_fj, weights[j], z)
            
            if tch_child < tch_current:
                current_solutions[j] = child
                current_f[j] = new_f


all_f = np.array([[f1(p), f2(p)] for p in current_solutions])

# Визуализация
plt.scatter(all_f[:, 0], all_f[:, 1], c='red', label='Фронт Парето')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('MOEA/D')
plt.grid(True)
plt.legend()
plt.show()
plt.show()
