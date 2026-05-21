"""
Project 1: NumPy Data Explorer
Syntecxhub Data Science Internship - Week 1
"""

import numpy as np
import time

print("=" * 60)
print("       PROJECT 1: NumPy Data Explorer")
print("=" * 60)

# ── 1. Array Creation ──────────────────────────────────────────
print("\n📌 1. ARRAY CREATION")
arr1d = np.array([10, 20, 30, 40, 50])
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
zeros  = np.zeros((3, 3))
ones   = np.ones((2, 4))
rng    = np.arange(0, 20, 2)
rand   = np.random.randint(1, 100, size=(4, 5))

print(f"1D array      : {arr1d}")
print(f"2D array:\n{arr2d}")
print(f"Zeros (3x3):\n{zeros}")
print(f"Ones  (2x4):\n{ones}")
print(f"Range (0-18, step 2): {rng}")
print(f"Random 4x5 array:\n{rand}")

# ── 2. Indexing & Slicing ──────────────────────────────────────
print("\n📌 2. INDEXING & SLICING")
print(f"arr1d[2]          : {arr1d[2]}")
print(f"arr2d[1][2]       : {arr2d[1][2]}")
print(f"arr1d[1:4]        : {arr1d[1:4]}")
print(f"arr2d[0, :]       : {arr2d[0, :]}")   # first row
print(f"arr2d[:, 1]       : {arr2d[:, 1]}")   # second column
print(f"arr2d[0:2, 0:2]   :\n{arr2d[0:2, 0:2]}")  # sub-matrix

# ── 3. Mathematical Operations ─────────────────────────────────
print("\n📌 3. MATHEMATICAL OPERATIONS")
a = np.array([1, 2, 3, 4, 5], dtype=float)
b = np.array([10, 20, 30, 40, 50], dtype=float)

print(f"a + b   : {a + b}")
print(f"b - a   : {b - a}")
print(f"a * b   : {a * b}")
print(f"b / a   : {b / a}")
print(f"a ** 2  : {a ** 2}")
print(f"sqrt(b) : {np.sqrt(b)}")

# ── 4. Axis-wise & Statistical Operations ─────────────────────
print("\n📌 4. STATISTICAL OPERATIONS")
dataset = np.random.randint(10, 100, size=(5, 4))
print(f"Dataset (5x4):\n{dataset}")
print(f"Mean (overall)   : {np.mean(dataset):.2f}")
print(f"Mean (col-wise)  : {np.mean(dataset, axis=0)}")
print(f"Mean (row-wise)  : {np.mean(dataset, axis=1)}")
print(f"Median           : {np.median(dataset):.2f}")
print(f"Std Dev          : {np.std(dataset):.2f}")
print(f"Min / Max        : {np.min(dataset)} / {np.max(dataset)}")
print(f"Sum (col-wise)   : {np.sum(dataset, axis=0)}")

# ── 5. Reshaping & Broadcasting ────────────────────────────────
print("\n📌 5. RESHAPING & BROADCASTING")
flat = np.arange(1, 13)
matrix = flat.reshape(3, 4)
print(f"Original (1x12): {flat}")
print(f"Reshaped (3x4):\n{matrix}")
print(f"Transposed (4x3):\n{matrix.T}")

# Broadcasting: add a row vector to each row of the matrix
row_vec = np.array([1, 2, 3, 4])
print(f"\nBroadcast add {row_vec} to each row:\n{matrix + row_vec}")

# ── 6. Save & Load ─────────────────────────────────────────────
print("\n📌 6. SAVE & LOAD OPERATIONS")
save_array = np.random.rand(3, 3)
np.save("/home/claude/saved_array.npy", save_array)
loaded = np.load("/home/claude/saved_array.npy")
print(f"Saved array:\n{save_array}")
print(f"Loaded array:\n{loaded}")
print(f"Arrays match: {np.allclose(save_array, loaded)}")

# Also save as CSV text
np.savetxt("/home/claude/saved_array.csv", save_array, delimiter=",", fmt="%.4f")
print("Saved to CSV as well.")

# ── 7. NumPy vs Python List Performance ───────────────────────
print("\n📌 7. NUMPY vs PYTHON LIST PERFORMANCE")
size = 1_000_000

# Python list
py_list = list(range(size))
start = time.time()
py_result = [x * 2 for x in py_list]
py_time = time.time() - start

# NumPy array
np_arr = np.arange(size)
start = time.time()
np_result = np_arr * 2
np_time = time.time() - start

print(f"Python list multiplication : {py_time:.4f} seconds")
print(f"NumPy array multiplication : {np_time:.4f} seconds")
print(f"NumPy is ~{py_time/np_time:.1f}x faster!")

print("\n✅ Project 1 Complete!")
