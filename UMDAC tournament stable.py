#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from interval import interval
import random
import statistics
import numpy as np
from scipy.stats import truncnorm


# In[2]:


# The input files follow the "Taillard" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Processing times for each job on each machine (given in the processing order)
    processing_times_in_processing_order = [[int(lines[i].split()[j]) for j in range(nb_machines)] for i in
                                            range(1, 1 + nb_jobs)]

    # Processing order of machines for each job
    machine_order = [[int(lines[i].split()[j]) - 1 for j in range(nb_machines)] for i in
                     range(1 + nb_jobs, 1 + 2 * nb_jobs)]

    # Reorder processing times: processing_time[j][m] is the processing time of the
    # activity of job j that is processed on machine m
    processing_time = [[processing_times_in_processing_order[j][machine_order[j].index(m)] for m in range(nb_machines)]
                       for j in range(nb_jobs)]

    # Trivial upper bound for the start times of the activities
    max_start = sum(sum(processing_time[j]) for j in range(nb_jobs))

    return (nb_jobs, nb_machines, processing_time, machine_order, max_start)


# In[3]:


def visualize(schedule):

    schedule = schedule.copy(deep=True)
    
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(JOBS)+len(MACHINES))/4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j,m) in schedule.index:
                xs = schedule.loc[(j,m), 'Start']
                xf = schedule.loc[(j,m), 'Finish']
                ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%7], **bar_style)
                ax[0].text((xs + xf)/2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%7], **bar_style)
                ax[1].text((xs + xf)/2, mdx, j, **text_style)
                
    ax[0].set_title('Job Schedule')
    ax[0].set_ylabel('Job')
    ax[1].set_title('Machine Schedule')
    ax[1].set_ylabel('Machine')
    
    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[1]+0.5, "Makespan: {0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)
        
    fig.tight_layout()


# In[4]:


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __repr__(self):
        return f'Chromsome: {self.chromosome}, Fitness: {self.fitness}'

    def calculate_fitness(self, df, nb_machines):
        feasible_solution = decode(self.chromosome, df, nb_machines)
        self.fitness = feasible_solution['Finish'].max()


# In[5]:


def is_overlap(interval_1, interval_2):
    union = interval_1 & interval_2
    if union == interval():
        return False
    if union[0][0] == union[0][1]:
        return False
    return True


# In[15]:


def decode(chromosome, df, nb_machines):
    ###
    # ORDNERANDO VECTOR D, PRIMERO POR DELAYS Y LUEGO POR JOB
    ###
    chromosome = sorted(chromosome, key=lambda item: (item[0], item[1]))
    #print(chromosome)

    ###
    # AÑADIR DELAYS A CADA TIEMPO DE INICIO Y FINALIZACIÓN
    ###
    df = df.copy(deep=True)

    for d, j in chromosome:
        df.loc[df['Job'] == j, 'Start'] += d
        df.loc[df['Job'] == j, 'Finish'] += d

    ###
    # TRATAR DE CALENDARIZAR DONDE DE ACUERDO A D
    ##
    # INICIALIZAR PERIODOS OCUPADOS DE CADA MÁQUINA
    busy_times = [interval()] * nb_machines
    # CALENARIZAR TAREA POR TAREA
    for _, j in chromosome:
        tmp = df[df['Job'] == j]
        delta = 0

        collision = False
        for _, operation in tmp.iterrows():
            machine = int(operation['Machine'])
            start = operation['Start']
            finish = operation['Finish']
            machine_busy_times = busy_times[machine]

            for component in machine_busy_times.components:
                if is_overlap(interval[start, finish], component) == True:
                    collision = True
                    break

        tmp = df[df['Job'] == j]
        if collision == False:
            for i, operation in tmp.iterrows():
                machine = int(operation['Machine'])
                start = operation['Start']
                finish = operation['Finish']
                busy_times[machine] = busy_times[machine] | interval([start, finish])
            continue

        ###
        # CALENDARIZAR LO MÁS CERCANO A 0
        ###
        d =  df.loc[df['Job'] == j]['Start'].min()
        df.loc[df['Job'] == j, 'Start'] -= d
        df.loc[df['Job'] == j, 'Finish'] -= d               

        posible_start_times = [0]

        tmp = df[df['Job'] == j]
        for _, operation in tmp.iterrows():
            machine = int(operation['Machine'])
            start = operation['Start']
            finish = operation['Finish']
            #duration = operation['Duration']
            machine_busy_times = busy_times[machine]

            upper_limit = 0
            for component in machine_busy_times.components:
                if start < component[0][1]:
                #upper_limit = component[0][1]
                #posible_start_times.append(upper_limit)
                    posible_start_times.append(component[0][1] - start)

        posible_start_times = list( dict.fromkeys(posible_start_times) )
        posible_start_times.sort()

        for st in posible_start_times:
            df.loc[df['Job'] == j, 'Start'] += st
            df.loc[df['Job'] == j, 'Finish'] += st

            collision = False
            tmp = df[df['Job'] == j]
            for _, operation in tmp.iterrows():
                machine = int(operation['Machine'])
                start = operation['Start']
                finish = operation['Finish']
                machine_busy_times = busy_times[machine]

                for component in machine_busy_times.components:
                    if is_overlap(interval[start, finish], component) == True:
                        collision = True

            if collision == False:
                delta = st
                break

            df.loc[df['Job'] == j, 'Start'] -= st
            df.loc[df['Job'] == j, 'Finish'] -= st
        
        tmp = df[df['Job'] == j]
        for i, operation in tmp.iterrows():
            machine = int(operation['Machine'])
            start = operation['Start']
            finish = operation['Finish']

            df.at[i, 'Start'] = start# + delta
            df.at[i, 'Finish'] = finish# + delta
            #busy_times[machine] = busy_times[machine] | interval([start + delta, finish + delta])
            busy_times[machine] = busy_times[machine] | interval([start, finish])

    if df['Start'].min() > 0:
        initial_gap = df['Start'].min()
        df['Start'] -= initial_gap
        df['Finish'] -= initial_gap
    return df


# In[7]:


def tournament_selection(population, k):
    parents = random.choices(population, k=k)
    parents = sorted(parents, key=lambda agent: agent.fitness, reverse=False)
    return parents[0]


# # 1. Leer el problema

# In[19]:


###
# LEER EL PROBLEMA
###
FILENAME = './la40.txt'

nb_jobs, nb_machines, processing_time, machine_order, max_start = read_instance(FILENAME)


# # 2. Obtener las cotas superiores de cada máquina

# In[20]:


# SUMATORIA DE TODAS LAS OPERACIONES DE CADA JOB
processing_times_per_job = [sum(processing_time[i]) for i in range(len(processing_time))]
# COTA SUPERIOR PARA CADA JOB
max_starts_of_jobs = list(map(lambda x:max_start-x, processing_times_per_job))


# # 3. Generar un schedule donde cada trabajo inicia en tiempo 0

# In[21]:


expanded_jobs = []
for j in range(nb_jobs):
    time = 0
    for k in range(nb_machines):
        machine = machine_order[j][k]
        expanded_jobs.append(
            {'Job': j,
            'Machine': machine,
            'Start': time, 
            'Duration': processing_time[j][machine], 
            'Finish': time + processing_time[j][machine]}
        )
        time += processing_time[j][machine]

df = pd.DataFrame(expanded_jobs)


# In[22]:


c = [(27, 0), (4, 1), (23, 2), (21, 3), (34, 4), (19, 5)]

c = sorted(c, key=lambda item: (item[0], item[1]))
df_choque = df.copy(deep=True)

for d, j in c:
    df_choque.loc[df['Job'] == j, 'Start'] += d
    df_choque.loc[df['Job'] == j, 'Finish'] += d

visualize(df_choque)


# In[23]:


c = [(6, 0), (23, 1), (24, 2), (45, 3), (46, 4), (7, 5)]
print(c)
df1 = decode(c, df, nb_machines)
visualize(df1)


# # 4. UMDAc

# In[25]:


GENERATIONS = 20
POP_SIZE = 60

# GENERANDO POBLACIÓN INICIAL
population = []
for _ in range(POP_SIZE):
    chromosome = [(random.randint(0, max_starts_of_jobs[i]), i) for i in range(nb_jobs)]
    ind = Individual(chromosome)
    population.append(ind)

for i in range(GENERATIONS):
    # POPULATION = CORRECCION DE COLISIONES(POPULATION)
    # POPULATION = CORRECION DE GAPS(POPULATION)
    new_pop = []
    for individual in population:
        df2 = decode(individual.chromosome, df, nb_machines)
        chromosome = []
        for j in range(nb_jobs):
            delay = df2[df['Job'] == j]['Start'].min()
            chromosome.append((delay, j))
        ind = Individual(chromosome)
        ind.fitness = df2['Finish'].max()
        new_pop.append(ind)
    
    bests = []
    for _ in range(POP_SIZE//2):
        ind = tournament_selection(new_pop, 5)
        bests.append(ind)

    bests.sort()
    print(bests[0].fitness)
    print(bests[0].chromosome)

    miu_dev = []
    
    for j in range(nb_jobs):
        vals = []
        for k in range(len(bests)):
            vals.append(bests[k].chromosome[j][0])
        miu = statistics.mean(vals)
        std = statistics.stdev(vals)
        miu_dev.append((miu, std))

    population = []
    for _ in range(POP_SIZE):
        #chromosome = [(int(truncnorm.rvs(a=0, b=max_start, loc=miu_dev[i][0], scale=miu_dev[i][1])), i) for i in range(nb_jobs)]
        chromosome = [(int(truncnorm.rvs(a=0, b=bests[0].fitness, loc=miu_dev[i][0], scale=miu_dev[i][1])), i) for i in range(nb_jobs)]
        ind = Individual(chromosome)
        population.append(ind)

