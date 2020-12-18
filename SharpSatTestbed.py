import subprocess
import sys
import os.path
import math
import time
import pickle
from pathlib import Path
import itertools
import random

import psutil, os

from functools import *
from operator import *

from multiprocessing import Process, Value

__VERBOSE__ = True
__DEBUG__ = False
__TIMEOUT__ = 10*60

def test_default_times(cnf_path_list):
    default_times_dict = {}

    for cnf_path in cnf_path_list:
        try:
            time = test_approxmc_single(cnf_path, timeout = __TIMEOUT__)

            default_times_dict[os.path.basename(cnf_path)] = time

            if __VERBOSE__:
                print(os.path.basename(cnf_path) + " (" + str(time) + "s)")
        except subprocess.TimeoutExpired:
            if __VERBOSE__:
                print(os.path.basename(cnf_path) + " (Benchmark timed out)")

                default_times_dict[os.path.basename(cnf_path)] = __TIMEOUT__

    return default_times_dict

def test_param_times(cnf_path_list, params_list):
    param_times_dict = {}

    for cnf_path in cnf_path_list:
        cnf_name = os.path.basename(cnf_path)
        for params in params_list:
            param_times_dict[(cnf_name, params)] = test_approxmc_single(cnf_path, params)

            if __VERBOSE__:
                print(str((cnf_name, params)) + " (" + str(param_times_dict[(cnf_name, params)]) + "s)")

    return param_times_dict

def optimize_params_list(cnf_path_list):
    optimized_params = {}

    for cnf_path in cnf_path_list:
        optimized_params[os.path.basename(cnf_path)] = optimize_params_single(cnf_path)

    return optimized_params

def create_portfolios(max_portfolio_size, cnf_path_list, params_list, param_times_dict):
    cnf_list = [os.path.basename(cnf_path) for cnf_path in cnf_path_list]

    portfolio_dict = {}

    for portfolio_size in range(1, max_portfolio_size+1):
        portfolio_combinations = itertools.combinations(params_list, portfolio_size)

        best_portfolio = []
        best_sum_time = float("inf")

        for current_portfolio in portfolio_combinations:
            current_sum_time = 0

            for cnf in cnf_list:
                best_cnf_time = float("inf")

                for params in current_portfolio:
                    best_cnf_time = min(param_times_dict[(cnf, params)], best_cnf_time)

                current_sum_time += best_cnf_time

            if current_sum_time < best_sum_time:
                best_sum_time = current_sum_time
                best_portfolio = current_portfolio

        if __VERBOSE__:
            print("Best Portfolio of size " + str(portfolio_size) + ":")
            print("Total Theoretical Time (" + str(best_sum_time) + "s)")

            for params in best_portfolio:
                print(params)

            print()

        portfolio_dict[portfolio_size] = (best_portfolio, best_sum_time)

    return portfolio_dict

def test_portfolio_times(cnf_path_list, portfolios_dict):
    portfolio_metrics = {}

    for portfolio_size in portfolios_dict.keys():
        portfolio = portfolios_dict[portfolio_size][0]

        for cnf_path in cnf_path_list:
            cnf = os.path.basename(cnf_path)

            portfolio_metrics[(cnf, portfolio_size)] = test_approxmc_parallel(cnf_path, portfolio, portfolio_size)

            if __VERBOSE__:
                print(cnf + ", P-Size=" + str(portfolio_size) + " (" + str(portfolio_metrics[(cnf, portfolio_size)]) + "s)")

    return portfolio_metrics

def optimize_params_single(cnf_path):
    best_time = __TIMEOUT__
    best_params = []

    search_space = []
    search_space.append([0,1])
    search_space.append([1,0])
    search_space.append([1,0])
    search_space.append([0,1])
    search_space.append([1,0])
    search_space.append([1.6,0.1,0.5,1.0,2.0,4.0])

    for sparse_val in search_space[0]:
        for deatchxor_val in search_space[1]:
            for reusemodels_val in search_space[2]:
                for forcesolextension_val in search_space[3]:
                    for simplify_val in search_space[4]:
                        for velimratio_val in search_space[5]:
                            params = (("sparse", sparse_val),  ("detachxor", deatchxor_val),  ("reusemodels", reusemodels_val), \
                                     ("forcesolextension", forcesolextension_val),  ("simplify", simplify_val),  ("velimratio", velimratio_val))

                            try:
                                curr_time = test_approxmc_single(cnf_path, params, timeout = best_time)
                            except subprocess.TimeoutExpired:
                                if __VERBOSE__: 
                                    print(str(params) + " (Params timed out, > " + str(best_time) + "s)")
                            else:
                                if __VERBOSE__: 
                                    print(str(params) + " (Params finished in " + str(curr_time) + "s)")

                            if curr_time < best_time:
                                best_time = curr_time
                                best_params = params

    return (best_params, best_time)



def test_approxmc_parallel(cnf_path, params_list, num_threads):
    """
    @param cnf_path: A string containing the path to the cnf
    @param params_list: A list of list of tuples containing (flag, arg) for "--flag arg".
            Each inner list is a param set for a different thread
    @param num_threads: Number of threads, must equal len(params)
    @return Time taken to solve cnf by the fastest thread
    """
    assert len(params_list) == num_threads

    thread_list = []
    best_time = None

    output_store = Value('d', -1.0)

    for params in params_list:
        thread = Process(target=test_approxmc_single, args=(cnf_path, params, output_store))
        thread.start()
        thread_list.append(thread)

    while True:
        time.sleep(0.1)
        if output_store.value != -1:
            best_time = output_store.value
            break

    for thread in thread_list:
        thread.terminate()

    return best_time

def test_approxmc_single(cnf_path, params = [], output_store = None, timeout = None):
    """
    @param cnf_path: A string containing the path to the cnf
    @param params: A list of tuples containing (flag, arg) for "--flag arg"
    @return Time taken to solve cnf
    """
    if __DEBUG__:
        print("Starting with ", params)

    arguments = ["approxmc", cnf_path]

    for param in params:
        arguments.append("--" + param[0])
        arguments.append(str(param[1]))

    process = subprocess.run(args=arguments, capture_output=True, check=True, timeout=timeout)

    output = process.stdout.decode("utf-8")

    unparsed_time = output.split("\n")[-4]

    time_str = unparsed_time.split()[-2]

    runtime = float(time_str)

    if __DEBUG__:
        print("Ending with ", params, " and total time ", runtime)

    if output_store is not None:
        with output_store.get_lock():
            if output_store.value == -1:
                output_store.value = runtime

    return runtime

if __name__ == '__main__':
    __RANDOM_SEED__ = 123456789
    __MAX_PORTFOLIO_SIZE__ = 4
    __MAX_TRAIN_TIME__ = 60
    __TRAIN_TEST_RATIO__ = 1

    __TRAIN_TEST_PICKLE__ = "data/train_test.pickle"
    __DEFAULT_TIMES_PICKLE__ = "data/default_times.pickle"
    __OPTIMIZED_PARAMS_PICKLE__ = "data/optimized_params.pickle"
    __PARAM_TIMES_PICKLE__ = "data/param_times.pickle"
    __PORTFOLIOS_PICKLE__ = "data/portfolios.pickle"
    __PORTFOLIO_METRICS_PICKLE__ = "data/portfolio_metrics.pickle"

    __DEFAULT_PARAMS__ = (("sparse", 0),  ("detachxor", 1),  ("reusemodels", 1),  ("forcesolextension", 0),  ("simplify", 1),  ("velimratio", 1.6))

    __FULL_BENCHMARK_DIR__ = "AnnotatedBenchmarks/"

    random.seed(__RANDOM_SEED__)

    full_benchmark_list = [str(path) for path in list(Path(__FULL_BENCHMARK_DIR__).rglob("*.cnf"))]
    full_benchmark_list.sort()

    print("Total Benchmarks: ", len(full_benchmark_list))

    #Creates mapping from CNF to it's path
    cnf_path_dict = {}

    for cnf_path in full_benchmark_list:
        cnf_path_dict[os.path.basename(cnf_path)] = cnf_path

    # Calculates/Loads Default Times
    # The time taken with default parameters for every benchmark possible

    if os.path.isfile(__DEFAULT_TIMES_PICKLE__):
        print("Loading execution times with default params...\n")
        default_times_dict = pickle.load(open(__DEFAULT_TIMES_PICKLE__, 'rb'))
    else:
        print("Calculating execution times with default params...\n")
        default_times_dict = test_default_times(full_benchmark_list)
        print()

    for cnf in default_times_dict:
        print(str(cnf) + " (" + str(default_times_dict[cnf]) + "s)")

    pickle.dump(default_times_dict, open(__DEFAULT_TIMES_PICKLE__, "wb"))

    print()

    # Calculates/Loads Train/Test Benchmark Allocations Times
    # Labels each benchmark as train or test

    if os.path.isfile(__TRAIN_TEST_PICKLE__):
        print("Loading train/test cnf benchmark allocation...\n")
        train_test_list = pickle.load(open(__TRAIN_TEST_PICKLE__, 'rb'))
    else:
        print("Calculating train/test cnf benchmark allocation...\n")

        train_benchmark_list = []
        test_benchmark_list = []

        #Complete initial categorization, with those simply too long to train with going into test
        for cnf in default_times_dict.keys():
            cnf_path = cnf_path_dict[cnf]
            if default_times_dict[cnf] > 1 and default_times_dict[cnf] < __TIMEOUT__:
                if default_times_dict[cnf] >__MAX_TRAIN_TIME__:
                    test_benchmark_list.append(cnf_path)
                else:
                    train_benchmark_list.append(cnf_path)

        print("Initial Train Benchmark List Length: ", len(train_benchmark_list))
        print("Initial Test Benchmark List Length: ", len(test_benchmark_list))
        print()

        while len(train_benchmark_list) >= __TRAIN_TEST_RATIO__*len(test_benchmark_list):
            pop_index = random.randrange(len(train_benchmark_list))

            print("Moving " + os.path.basename(train_benchmark_list[pop_index]) + " from Train to Test")

            test_benchmark_list.append(train_benchmark_list.pop(pop_index))
        print()

        print("Final Train Benchmark List Length: ", len(train_benchmark_list))
        print("Final Test Benchmark List Length: ", len(test_benchmark_list))
        print()

        train_test_list = [train_benchmark_list, test_benchmark_list]

    train_benchmark_list = train_test_list[0]
    test_benchmark_list = train_test_list[1]
    active_benchmark_list = train_benchmark_list + test_benchmark_list

    train_benchmark_list.sort()
    test_benchmark_list.sort()
    active_benchmark_list.sort()

    print("Full Benchmark List:")
    print("".join([path + "\n" for path in full_benchmark_list]), "\n")

    print("Train Benchmark List:")
    print("".join([path + "\n" for path in train_benchmark_list]), "\n")

    print("Test Benchmark List:")
    print("".join([path + "\n" for path in test_benchmark_list]), "\n")

    pickle.dump(train_test_list, open(__TRAIN_TEST_PICKLE__, "wb"))

    # Calculates/Loads Optimized Params
    # The optimal parameters based off a grid search for each benchmark in the train set.

    if os.path.isfile(__OPTIMIZED_PARAMS_PICKLE__):
        print("Loading optimized parameters...\n")
        optimized_params_dict = pickle.load(open(__OPTIMIZED_PARAMS_PICKLE__, 'rb'))
    else:
        print("Calculating optimal parameters...\n")
        optimized_params_dict = optimize_params_list(train_benchmark_list)
        print()

    for cnf in optimized_params_dict:
        print(cnf, ": ", optimized_params_dict[cnf])

    pickle.dump(optimized_params_dict, open(__OPTIMIZED_PARAMS_PICKLE__, "wb"))
    print()

    #Prints the speedup that was achieved by optimizing parameters for each benchmark in the train set.

    print("Tuned CNF Speedups...\n")

    for cnf_path in train_benchmark_list:
        cnf = os.path.basename(cnf_path)
        print(cnf + ": (" + str(default_times_dict[cnf]) + "s) -> (" + str(optimized_params_dict[cnf][1]) + "s)")

    print()

    # Creates new list of parameters without duplicates

    print("Calculating minimal parameter set...\n")

    condensed_params = set()

    for param in optimized_params_dict.values():
        if tuple(param[0]) not in condensed_params:
            condensed_params.add(tuple(param[0]))

    condensed_params = list(condensed_params)


    print("Condensed params: " + str(len(optimized_params_dict.values())) + " -> " + str(len(condensed_params)))

    for param in condensed_params:
        print(param)

    print()


    # Calculates/Loads Param Times
    # The amount of time each parameter takes on each benchmark in the training set

    if os.path.isfile(__PARAM_TIMES_PICKLE__):
        print("Loading parameter times on all benchmarks...\n")
        param_times_dict = pickle.load(open(__PARAM_TIMES_PICKLE__, 'rb'))
    else:
        print("Calculating parameter times on all benchmarks ...\n")
        param_times_dict = test_param_times(train_benchmark_list, condensed_params)
        print()

    for test_pair in param_times_dict.keys():
        print(str(test_pair[0]) + " " + str(test_pair[1]) + " (" + str(param_times_dict[test_pair]) + "s)")

    print()

    pickle.dump(param_times_dict, open(__PARAM_TIMES_PICKLE__, "wb"))


    # Calculates/Loads Optimal Portfolios
    # The optimal portfolio based off the param times for each portfolio size

    if os.path.isfile(__PORTFOLIOS_PICKLE__):
        print("Loading optimal portfolios...\n")
        portfolios_dict = pickle.load(open(__PORTFOLIOS_PICKLE__, 'rb'))
    else:
        print("Calculating optimal portfolios ...\n")
        portfolios_dict = create_portfolios(__MAX_PORTFOLIO_SIZE__, train_benchmark_list, condensed_params, param_times_dict)
        print()


    training_default_sum_time = 0
    for cnf_path in train_benchmark_list:
        cnf = os.path.basename(cnf_path)
        training_default_sum_time += default_times_dict[cnf]

    print("Default Parameters Sum Time on Training CNFs (" + str(training_default_sum_time) + "s)\n")

    for portfolio_size in range(1, __MAX_PORTFOLIO_SIZE__+1):
        portfolio_data = portfolios_dict[portfolio_size]
        print("Best Portfolio of size " + str(portfolio_size) + ":")
        print("Total Theoretical Time on Training CNFs (" + str(portfolio_data[1]) + "s)")

        for params in portfolio_data[0]:
            print(params)

        print()

    print()

    pickle.dump(portfolios_dict, open(__PORTFOLIOS_PICKLE__, "wb"))


    # Calculates/Loads Portfolio metrics
    # A measurement of how well each portfolio does on each benchmark in training and test

    if os.path.isfile(__PORTFOLIO_METRICS_PICKLE__):
        print("Loading portfolio metrics...\n")
        portfolio_metrics = pickle.load(open(__PORTFOLIO_METRICS_PICKLE__, 'rb'))
    else:
        print("Calculating portfolio metrics...\n")
        portfolio_metrics = test_portfolio_times(active_benchmark_list, portfolios_dict)
        print()

    for portfolio_size in portfolios_dict.keys():
        print("Metrics for Size " + str(portfolio_size) + " Portfolio:")

        for cnf_path in active_benchmark_list:
            cnf = os.path.basename(cnf_path)
            print(cnf + ": (" + str(portfolio_metrics[(cnf, portfolio_size)]) + "s)")

        print()
    print()

    pickle.dump(portfolio_metrics, open(__PORTFOLIO_METRICS_PICKLE__, "wb"))


    # Prints out theoretical and actual speedups for the train and testbenchmarks
    print("Default Parameters Sum Time on Training CNFs (" + str(training_default_sum_time) + "s)")

    # Prints out theoretical and actual speedups for the test benchmarks
    test_default_sum_time = 0
    for cnf_path in test_benchmark_list:
        cnf = os.path.basename(cnf_path)
        test_default_sum_time += default_times_dict[cnf]

    print("Default Parameters Sum Time on Test CNFs (" + str(test_default_sum_time) + "s)\n")

    for portfolio_size in range(1, __MAX_PORTFOLIO_SIZE__+1):
        portfolio_data = portfolios_dict[portfolio_size]
        print("Best Portfolio of size " + str(portfolio_size) + ":")
        print("Total Theoretical Time on Training CNFs (" + str(portfolio_data[1]) + "s)")

        train_total_actual_time = 0
        for cnf_path in train_benchmark_list:
            cnf = os.path.basename(cnf_path)
            train_total_actual_time += portfolio_metrics[(cnf, portfolio_size)]

        test_total_actual_time = 0
        for cnf_path in test_benchmark_list:
            cnf = os.path.basename(cnf_path)
            test_total_actual_time += portfolio_metrics[(cnf, portfolio_size)]

        train_average_actual_speedup = 1
        for cnf_path in train_benchmark_list:
            cnf = os.path.basename(cnf_path)
            individual_speedup = default_times_dict[cnf]/portfolio_metrics[(cnf, portfolio_size)]
            train_average_actual_speedup *= individual_speedup
        train_average_actual_speedup = math.pow(train_average_actual_speedup, 1/len(train_benchmark_list))

        test_average_actual_speedup = 1
        for cnf_path in test_benchmark_list:
            cnf = os.path.basename(cnf_path)
            individual_speedup = default_times_dict[cnf]/portfolio_metrics[(cnf, portfolio_size)]
            test_average_actual_speedup *= individual_speedup
        test_average_actual_speedup = math.pow(test_average_actual_speedup, 1/len(test_benchmark_list))

        print("Actual Total Time on Training CNFs: (" + str(training_default_sum_time) + "s) -> (" + str(train_total_actual_time) + "s)")

        print("Theoretical Total Train Speedup: " + str((portfolio_data[1]/training_default_sum_time)**-1))
        print("Actual Total Train Speedup: " + str((train_total_actual_time/training_default_sum_time)**-1))
        print("Actual Average Train Speedup: " + str(train_average_actual_speedup))

        print("Actual Total Time on Test CNFs: (" + str(test_default_sum_time) + "s) -> (" + str(test_total_actual_time) + "s)")

        print("Actual Total Test Speedup: " + str((test_total_actual_time/test_default_sum_time)**-1))
        print("Actual Average Test Speedup: " + str(test_average_actual_speedup))

        for params in portfolio_data[0]:
            print(params)

        print()

    print()

