import os, sys

current_dir = os.path.join(os.getcwd(), 'python')
sys.path.append(current_dir)

import needle as ndl

a = ndl.ones(32).reshape((4,2,4))
print(a)

ndl.ops.LogSumExp(axes = (2))(a)