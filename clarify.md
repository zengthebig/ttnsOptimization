明确模型: 每一层都按照相关性分成若干个TTNS, 每个TTNS结构由chow-liu树给出

假设上一层是X_j, 下一层是Y_i, Y_i = \max\limits_{j\in S_i}(X_j+ D_{ij})

有: F_Y(y) = \int P_X(x)\prod_{i=1}^{n1}\prod_{j\in S_i} F_{ij}(y_i-x_j)dx

代入P_X(x)的TTNS结构, 可以化简得到F_Y(y)的表达式的. 