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