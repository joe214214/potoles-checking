def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


if __name__ == "__main__":
    test_cases = [
        [3, 6, 8, 10, 1, 2, 1],
        [],
        [1],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [42],
        [3, 3, 3, 1, 1, 2],
    ]

    for arr in test_cases:
        result = quicksort(arr)
        print(f"Input:  {arr}")
        print(f"Output: {result}")
        print()