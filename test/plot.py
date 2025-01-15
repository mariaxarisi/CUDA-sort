import os
import matplotlib.pyplot as plt

FILE_PATH = os.path.join(os.path.dirname(__file__), 'GTX750.txt')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../assets')
SAVE_PATH = os.path.join(SAVE_DIR, 'GTX750.png')

data  = {'N': [], 'qsort': [], 'V0': [], 'V1': [], 'V2': []}

# read and parse the data from GTX750.txt
with open(FILE_PATH, 'r') as file:
    curr_N = None
    prev_line = ''
    for line in file:
        line = line.strip()
        if line.startswith('N = '):
            curr_N = int(line.split('=')[1].strip())
            data['N'].append(curr_N)
        elif line.startswith('Execution Time:'):
            exec_time = line.split(': ')[1].strip()
            if exec_time != 'NULL sec':
                exec_time = float(exec_time.split()[0])
            else:
                exec_time = None
            if 'qsort' in prev_line:
                data['qsort'].append(exec_time)
            elif 'V0' in prev_line:
                data['V0'].append(exec_time)
            elif 'V1' in prev_line:
                data['V1'].append(exec_time)
            elif 'V2' in prev_line:
                data['V2'].append(exec_time)
        prev_line = line

# plot performance
plt.figure(figsize=(10, 6))
plt.plot(data['N'], data['qsort'], label='qsort', marker='o')
plt.plot(data['N'], data['V0'], label='V0', marker='o')
plt.plot(data['N'], data['V1'], label='V1', marker='o')
plt.plot(data['N'], data['V2'], label='V2', marker='o')

plt.xlabel('N')
plt.ylabel('Execution Time (sec)')
plt.title('NVIDIA GeForce GTX 750 Performance')
plt.grid(True)
plt.legend()

plt.savefig(SAVE_PATH)
plt.show()