import timeit
from collections import deque

# Initialize the original list
original_list = list(range(100))
m = 1000  # Number of insertions

# Method using list.insert(0, value)
def list_insert():
    lst = original_list[:]
    for _ in range(m):
        lst.insert(0, _)

# Method using deque
def deque_insert():
    deq = deque(original_list)
    for _ in range(m):
        deq.appendleft(_)
    lst = list(deq)
    lst.append(1)

# Time the operations
time_list_insert = timeit.timeit(list_insert, number=100)
time_deque_insert = timeit.timeit(deque_insert, number=100)

print("Time with list insert:", time_list_insert)
print("Time with deque insert:", time_deque_insert)

# def main() -> None:
#     # custom = np.array([1., 1., 0.0]).reshape((1,3))
#     # grid_points = generate_grid_points([-2,2,-2,2])
#     # custom = np.concatenate((custom, grid_points),axis=0)
#     # print(f"{custom=}\n{grid_points=}")
#     # for i in custom:
#     #     print(f"{i[0]=}")

#     bounds = np.array([-2,2,-2,2])
#     sizes = bounds[1::2] - bounds[::2]
#     diagonal = np.sqrt(np.sum(sizes ** 2))
#     step_size = diagonal / (100 * np.sqrt(np.mean(sizes ** 2)))
#     print(step_size)

# if __name__ == '__main__':
#     main()