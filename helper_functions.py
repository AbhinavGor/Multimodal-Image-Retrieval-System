
def find_lowest_index_greater_than(arr, target_value):
    left, right = 0, len(arr) - 1
    lowest_index = None

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] > target_value:
            lowest_index = mid  # Update the lowest index found so far
            right = mid - 1
        else:
            left = mid + 1

    return lowest_index