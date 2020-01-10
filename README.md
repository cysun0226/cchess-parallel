cchess-parallel
=

The serial version source code is modified from [cchess-zero](https://github.com/chengstone/cchess-zero).

Based on the serial version, we implement the root parallelization of MCTS by three methods: multi-process module, multi-process with pool, and mpi4py.

# Porject Files Overview
```
.
├── ChessBoard.py
├── ChessGame.py
├── chessman                 // python file for each kind of chess
├── ChessPiece.py
├── ChessView.py
├── docker-compose.yml       // docker-compose config file for MPI experiment
├── hostfile                 // MPI hostfile of docker experiment
├── images                   // Image of chess (for UI)
├── main.py                  // serial version
├── main_mpi.py              // MPI source code
├── main_multiproc.py        // multi-process 
├── main_pool.py             // multi-process with pool
└── policy_value_network.py  // evaluation function of the new expanded node
```

# Execution

All the main programs (main.py, main_mpi.py, main_multiproc.py, main_pool.py) can be executed in the following command format. Users can adjust the  ```play_playout```, which specifies the number of game tree searching for each step.

### Playwith AI
```
python3 main.py --mode play --ai_count 1 --ai_function mcts --play_playout 5
```

### Watch AI vs AI
```
python3 main.py --mode play --ai_count 2 --ai_function mcts --play_playout 5
```

### Calculate MCTS time
This mode will not show GUI, and it will execute MCTS for the given times. The number of the iteration is specified by ```mcts_test_time```.
```
python3 main.py --train_playout 5 --mcts_test_time 5
```

# Source code overview

In this section, to quickly illustrate the key points of our implementation, we extract the segments related to the parallelization from our source code.

## mpi4py (main_mpi.py)
Most of the part of the parallelization is line 468~528, which shows in the following code section. The part of the master broadcasting the next action is in the function *getaction*, and the exact location starts from the line 1445.
```python
# make sure all the nodes complete updating
global comm
global MPI_size
global total_mcts_count
comm.Barrier()

# print("rank [", rank, "] reach the barrier")
if (rank == 0):
    print("\nround", total_mcts_count, "start\n")
sys.stdout.flush()

# if master, calculate the time
if rank == 0:
    start = timeit.default_timer()

# seperate tasks
# (omitted)

for i in range(MPI_size):
    result_list.append({})
if rank != MPI_size -1:
    child_list[rank] = dict(node_child_items[part*i:part*(i+1)])
else:
    child_list[rank] = dict(node_child_items[part*i:len(node.child)])

# search by each process
rand = ""
for c in child_list[rank]:
    rand = c
    for n in range(playouts):
        value = self.start_child_search(node, c, node.child[c], current_player, restrict_round)
        node.child[c].back_up_value(value)
compute_time = 0

if rank == 0:
    stop = timeit.default_timer()
    compute_time = stop - start
    print('Compute time: ', stop - start, "s")
    trans_start = timeit.default_timer()

# collect child from each parallel process
# send each child's node.v and the tree to the main process

# if master
# collect child from each parallel process
if rank == 0:
    for i in range(1, MPI_size):
        child_list[i] = comm.recv(source=i,tag=i)
else:
    my_child = {}
    for c in child_list[rank]:
        my_child[c] = child_list[rank][c]
    comm.send(my_child, dest=0, tag=rank)

# master update child according to the collected data
for i in range(1, MPI_size):
    for c in child_list[i]:
        node.child[c] = child_list[i][c]
```

## multiprocess (main_multiproc.py)
Adopt from line 461-511.
```python
# seperate children
for i in range(process_num):
    if i != process_num - 1:
        child_list.append(dict(node_child_items[part * i:part * (i + 1)]))
    else:
        child_list.append(dict(node_child_items[part * i:len(node.child)]))

# search by each process
q0 = mp.Queue()
q1 = mp.Queue()
q2 = mp.Queue()
q3 = mp.Queue()

process0 = mp.Process(target=self.search_by_each_child, args=(q0, child_list[0], playouts, node, current_player, restrict_round))
process1 = mp.Process(target=self.search_by_each_child, args=(q1, child_list[1], playouts, node, current_player, restrict_round))
process2 = mp.Process(target=self.search_by_each_child, args=(q2, child_list[2], playouts, node, current_player, restrict_round))
process3 = mp.Process(target=self.search_by_each_child, args=(q3, child_list[3], playouts, node, current_player, restrict_round))

process0.start()
process1.start()
process2.start()
process3.start()

# collect child from each parallel process
tmp = []
tmp.append(q0.get())
tmp.append(q1.get())
tmp.append(q2.get())
tmp.append(q3.get())
for i in range(4):
    for c in child_list[i]:
        node.child[c] = tmp[i].child[c]

# wait child processes finish
process0.join()
process1.join()
process2.join()
process3.join()

```

## multiprocess with pool (main_pool.py)
Adopt from line 420-428, 463-509.
```python
def search_by_each_child(self,child_list, playouts, node, current_player, restrict_round):
    #print('search_by_each_child')
    for c in child_list:
        for n in range(playouts):
            value = self.start_child_search(node, c, node.child[c], current_player, restrict_round)
            node.child[c].back_up_value(value)
    for c in child_list:
            self.node.child[c] = node.child[c]
    return node


# (main) seperate children
process_num = 4
child_list = []
ori_child = node.child.copy()
part = int(len(node.child) / process_num)

node_child_items = [x for x in node.child.items()]
node_child_items = sorted(node_child_items, key=lambda tup: tup[0])
        
for i in range(process_num):
    if i != process_num - 1:
        child_list.append(dict(node_child_items[part * i:part * (i + 1)]))
    else:
        child_list.append(dict(node_child_items[part * i:len(node.child)]))

# search by each process
PROCESSES = 4
data = ([child_list[0], playouts, node, current_player, restrict_round],[child_list[1], playouts, node, current_player, restrict_round],[child_list[2], playouts, node, current_player, restrict_round],[child_list[3], playouts, node, current_player, restrict_round])
pool = Pool(PROCESSES)
result = [pool.apply_async(self.search_by_each_child,(child_list[i], playouts, node, current_player, restrict_round)) for i in range(PROCESSES)]
pool.close()
pool.join()
pool.terminate()
```



