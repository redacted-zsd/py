
# Algorithm Patterns for Infrastructure Security

## 1. Prefix Sum
- **Use Case**: Analyzing network traffic over time to find periods with the highest volume of data transfer.
- **How to Implement**:
  ```python
  def prefix_sum(arr):
      prefix = [0] * (len(arr) + 1)
      for i in range(1, len(arr) + 1):
          prefix[i] = prefix[i - 1] + arr[i - 1]
      return prefix
  ```
- **Benefits**: Fast range queries.
- **Trade-offs**: High initial computation cost for large arrays.

---

## 2. Two Pointer
- **Use Case**: Detecting bandwidth overuse by analyzing traffic logs from both ends of the time range.
- **How to Implement**:
  ```python
  def two_pointer(arr, target):
      left, right = 0, len(arr) - 1
      while left < right:
          total = arr[left] + arr[right]
          if total == target:
              return left, right
          elif total < target:
              left += 1
          else:
              right -= 1
  ```
- **Benefits**: Efficient for sorted data.
- **Trade-offs**: Only works with sorted arrays or intervals.

---

## 3. Sliding Window
- **Use Case**: Monitoring failed login attempts over a time window to detect brute-force attacks.
- **How to Implement**:
  ```python
  def sliding_window(arr, k):
      max_sum, current_sum = 0, 0
      for i in range(k):
          current_sum += arr[i]
      max_sum = current_sum
      for i in range(k, len(arr)):
          current_sum += arr[i] - arr[i - k]
          max_sum = max(max_sum, current_sum)
      return max_sum
  ```
- **Benefits**: Efficient for continuous monitoring.
- **Trade-offs**: Window size must be predefined.

---

## 4. Fast & Slow Pointer
- **Use Case**: Detecting cycles in monitoring logs (e.g., repeated IP activity).
- **How to Implement**:
  ```python
  def has_cycle(logs):
      slow, fast = 0, 0
      while fast < len(logs) and fast + 1 < len(logs):
          slow += 1
          fast += 2
          if logs[slow] == logs[fast]:
              return True
      return False
  ```
- **Benefits**: Detects cycles efficiently.
- **Trade-offs**: Only useful for cyclic detection.

---

## 5. Linked List In-Place Reversal
- **Use Case**: Reversing the order of IP filtering rules for prioritized access control.
- **How to Implement**:
  ```python
  def reverse_linked_list(head):
      prev, current = None, head
      while current:
          next_node = current.next
          current.next = prev
          prev = current
          current = next_node
      return prev
  ```
- **Benefits**: In-place and memory efficient.
- **Trade-offs**: Works only for linked lists.

---

## 6. Monotonic Stack
- **Use Case**: Tracking increasing or decreasing patterns in CPU or memory usage to detect abnormal spikes.
- **How to Implement**:
  ```python
  def monotonic_stack(arr):
      stack = []
      for val in arr:
          while stack and stack[-1] < val:
              stack.pop()
          stack.append(val)
      return stack
  ```
- **Benefits**: Efficient for range queries.
- **Trade-offs**: Only works with specific patterns.

---

## 7. Top 'k' Elements
- **Use Case**: Identifying the top `k` IP addresses generating the most traffic in a specific time period.
- **How to Implement**:
  ```python
  import heapq

  def top_k_elements(arr, k):
      return heapq.nlargest(k, arr)
  ```
- **Benefits**: Efficient for finding top elements.
- **Trade-offs**: More complex than a simple sort.

---

## 8. Quick Select
- **Use Case**: Finding the median CPU usage for real-time load balancing.
- **How to Implement**:
  ```python
  def quick_select(arr, k):
      pivot = arr[len(arr) // 2]
      lows = [x for x in arr if x < pivot]
      highs = [x for x in arr if x > pivot]
      pivots = [x for x in arr if x == pivot]
      
      if k < len(lows):
          return quick_select(lows, k)
      elif k < len(lows) + len(pivots):
          return pivots[0]
      else:
          return quick_select(highs, k - len(lows) - len(pivots))
  ```
- **Benefits**: Efficient for finding specific ranks.
- **Trade-offs**: May be complex for large datasets.

---

## 9. Overlapping Intervals
- **Use Case**: Merging overlapping IP address ranges in firewall configurations.
- **How to Implement**:
  ```python
  def merge_intervals(intervals):
      intervals.sort(key=lambda x: x[0])
      merged = [intervals[0]]
      for current in intervals[1:]:
          last = merged[-1]
          if current[0] <= last[1]:
              last[1] = max(last[1], current[1])
          else:
              merged.append(current)
      return merged
  ```
- **Benefits**: Helps manage overlapping ranges.
- **Trade-offs**: Sorting may be costly for large datasets.

---

## 10. Modified Binary Search
- **Use Case**: Finding a security breach in sorted logs of access times.
- **How to Implement**:
  ```python
  def binary_search(arr, target):
      left, right = 0, len(arr) - 1
      while left <= right:
          mid = (left + right) // 2
          if arr[mid] == target:
              return mid
          elif arr[mid] < target:
              left = mid + 1
          else:
              right = mid - 1
      return -1
  ```
- **Benefits**: Fast search on sorted data.
- **Trade-offs**: Limited to sorted or rotated arrays.

---

## 11. Depth-First Search (DFS)
- **Use Case**: Exploring all paths in a network to identify security vulnerabilities.
- **How to Implement**:
  ```python
  def dfs(graph, node, visited):
      if node not in visited:
          visited.add(node)
          for neighbor in graph[node]:
              dfs(graph, neighbor, visited)
  ```
- **Benefits**: Efficient for exploring deep structures.
- **Trade-offs**: May not find the shortest path.

---

## 12. Breadth-First Search (BFS)
- **Use Case**: Finding the shortest path to block an attack on a network.
- **How to Implement**:
  ```python
  from collections import deque

  def bfs(graph, start):
      visited, queue = set(), deque([start])
      while queue:
          node = queue.popleft()
          if node not in visited:
              visited.add(node)
              queue.extend(graph[node] - visited)
      return visited
  ```
- **Benefits**: Finds shortest paths.
- **Trade-offs**: Memory-intensive for large graphs.

---

## 13. Matrix Traversal
- **Use Case**: Inspecting a grid of network nodes to find security issues.
- **How to Implement**:
  ```python
  def traverse_matrix(matrix):
      for row in matrix:
          for elem in row:
              print(elem)
  ```
- **Benefits**: Easy for grid-based problems.
- **Trade-offs**: May be inefficient for large matrices.

---

## 14. Backtracking
- **Use Case**: Finding all possible security configurations for firewall rules.
- **How to Implement**:
  ```python
  def backtrack(path, choices):
      if not choices:
          return path
      for choice in choices:
          new_choices = choices - set([choice])
          backtrack(path + [choice], new_choices)
  ```
- **Benefits**: Explores all possibilities.
- **Trade-offs**: Can be slow for large problems.

---

## 15. Dynamic Programming
- **Use Case**: Optimizing load balancing by breaking down a large problem into smaller subproblems.
- **How to Implement**:
  ```python
  def dynamic_programming(n):
      dp = [0] * (n + 1)
      dp[1] = 1
      for i in range(2, n + 1):
          dp[i] = dp[i - 1] + dp[i - 2]
      return dp[n]
  ```
- **Benefits**: Efficient for overlapping subproblems.
- **Trade-offs**: May use extra memory.

---
