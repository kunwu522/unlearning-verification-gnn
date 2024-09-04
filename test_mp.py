import random
from functools import partial
import multiprocessing
import time

def worker_process(process_id, num_tried):
        print(f"Process {process_id} is running...")
        time.sleep(10)
        rand_cond = random.choice([0, 1, 2, 3])
        if rand_cond:
             return process_id
        else:
             return None



def main():
    num_processes = 5
    # perturbations = multiprocessing.Value('i', [])  # Shared integer for the condition
    # shared_condition = multiprocessing.Value('i', 0)

    with multiprocessing.Pool(num_processes) as pool:
        perturbations = multiprocessing.Manager().list()
        num_tried = multiprocessing.Value('i', 0)
        for x in pool.imap_unordered(
            partial(worker_process, num_tried=num_tried, perturbations=perturbations),
            range(num_processes),
        ):
            if x is not None:
                perturbations.append(x)
            if len(perturbations) == 10 or num_tried == 20:
                break
    # Run the processes until a certain condition is met
    # time.sleep(10)  # Replace this with your actual condition check

    # Set the condition to True to signal the processes to stop
    # if len(perturbations) == 10 or num_tried == 20:
    #     shared_condition.value = 1

    # Wait for all processes to finish
    print(perturbations)

    print("All processes finished, perturbations: ", perturbations, num_tried)

if __name__ == "__main__":
    main()