
from gcproc import GCProc
import numpy as np
import matplotlib.pyplot as plt
import crocodile.toolbox as tb

"""

"""

super_dataset = []  # by convention, the target dataset comes last in this list.
gcs = [GCProc() for _ in range(len(super_dataset))]


iter_num = 15
for _ in range(iter_num):
    for index in range(len(gcs)):
        curr_gc = gcs[index]
        prev_gc = gcs[index - 1] if index > 0 else gcs[0]
        curr_gc.fit(data_list=super_dataset[index], code=prev_gc.code, encode=prev_gc.encode)


if __name__ == '__main__':
    pass
