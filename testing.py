import numpy as np
# import timeit
# from collections import deque

# # Initialize the original list
# original_list = list(range(100))
# m = 1000  # Number of insertions

# # Method using list.insert(0, value)
# def list_insert():
#     lst = original_list[:]
#     for _ in range(m):
#         lst.insert(0, _)

# # Method using deque
# def deque_insert():
#     deq = deque(original_list)
#     for _ in range(m):
#         deq.appendleft(_)
#     lst = list(deq)
#     lst.append(1)

# # Time the operations
# time_list_insert = timeit.timeit(list_insert, number=100)
# time_deque_insert = timeit.timeit(deque_insert, number=100)

# print("Time with list insert:", time_list_insert)
# print("Time with deque insert:", time_deque_insert)

def main() -> None:
    lst = np.array([1,2,3,4])
    zweidrei = lst[[1,2]]
    print(zweidrei)

if __name__ == '__main__':
    main()