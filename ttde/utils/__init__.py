"""
这个文件是一堆“工具函数/小零件”，专门给 JAX 项目用的。
你可以把它理解成：为了让训练/计算更快、更方便，写的一些通用辅助工具。

包含的功能：
1) 生成带日期时间的文件夹路径（方便保存实验结果，不会覆盖）
2) 把 einsum（张量缩并）做缓存（同样的表达式+shape，下次直接复用计划，速度更快）
3) 把一堆“树结构参数”(pytree) 按叶子逐个 stack（方便 vmap 批处理）
4) 一个秒表 Stopwatch（测时间用）
5) index(...)：让你可以像 batched_struct[i] 一样，从批量 pytree 里取第 i 个样本
"""

import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Union

import jax
from opt_einsum import contract_expression
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from jax import tree_util

def suffix_with_date(folder: Path) -> Path:
    """
    给一个文件夹路径加上当前日期时间作为子目录名，返回新路径。

    用途：
    - 你每次跑实验都想保存结果到一个新目录
    - 不想手动改名字、也不想覆盖之前的结果

    示例：
    folder = Path("runs")
    返回类似：runs/2026-02-06_20:15:33.123456
    """
    # 如果传进来的不是 Path（比如是字符串），就转换成 Path
    if type(folder) is not Path:
        folder = Path(folder)

    # datetime.now() 会得到当前时间，转成字符串后把空格替换成下划线
    # 原来可能是 "2026-02-06 20:15:33.123456"
    # 替换后变成 "2026-02-06_20:15:33.123456"
    now = str(datetime.now()).replace(' ', '_')

    # 返回 folder / now（等价于拼路径）
    return folder / now


# PathType 是一个类型别名：
# 你可以传字符串路径，也可以传 Path 对象
PathType = Union[str, Path]


def cached_einsum(expr: str, *args):
    """
    这是一个“加速版的 einsum”。

    你可以把 einsum 想成：给我一个缩并公式 expr（比如 "ij,jk->ik"）
    再给我一些张量 args，然后算结果。

    为什么要 cached？
    - opt_einsum 会先为这个 expr + 各个输入的 shape 计算出一个“最优缩并路线”（计划）
    - 这个计划如果每次都重新算，会浪费时间
    - 所以我们用缓存：同样 expr + 同样 shape，下次直接复用

    用法示例：
    y = cached_einsum("ij,jk->ik", A, B)
    """
    # 先根据 expr 和每个输入张量的 shape，拿到一个“缩并计划函数”
    # 然后把实际张量传进去执行，backend='jax' 表示用 JAX 进行计算
    return cached_einsum_expr(expr, *[arg.shape for arg in args])(*args, backend='jax')


@lru_cache
def cached_einsum_expr(expr: str, *shapes):
    """
    真正做缓存的地方（lru_cache 是 Python 自带的缓存装饰器）。

    输入：
      expr: einsum 的公式字符串
      shapes: 每个输入张量的 shape（比如 (32,64), (64,128) ...）

    输出：
      一个可调用对象（缩并计划），之后你可以像函数一样喂张量进去算。

    重点：
    - 这里缓存的是“路线/计划”，不是具体数值
    - 所以只要 expr + shapes 相同，就能复用
    """
    return contract_expression(expr, *shapes)


def tree_stack(trees):
    """
    给你一堆结构完全相同的“树结构”(pytree)，把它们按叶子逐个 stack 起来。

    你可以把 pytree 理解成：
      - 嵌套结构：dict / list / tuple / dataclass 里面放着数组
      - JAX 很喜欢用这种结构存参数

    举个最简单的例子：
      trees = [
        ((a, b), c),
        ((a2, b2), c2),
      ]
    返回：
      ((stack([a,a2]), stack([b,b2])), stack([c,c2]))

    用途：
    - 你有一堆“同结构参数/样本”
    - 想把它们变成一个“批量结构”，方便喂给 jax.vmap 批量计算
    """
    leaves_list = []   # 用来存每棵树的叶子（叶子=最底层的数组）
    treedef_list = []  # 用来存每棵树的结构信息（形状/嵌套方式）

    # 遍历每棵树：把它“拍扁”成叶子列表 + 结构定义
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    # grouped_leaves 会把“同一位置的叶子”放到一起
    # 例如第一棵树的第0个叶子、第二棵树的第0个叶子 -> 变成一个组
    grouped_leaves = zip(*leaves_list)

    # 对每一组叶子做 jnp.stack，得到批量维度
    # 例如两个 shape=(d,) 的叶子 stack 后变成 shape=(2,d)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]

    # 用第一棵树的结构 treedef，把这些批量叶子重新装回原来的嵌套结构
    return treedef_list[0].unflatten(result_leaves)


@dataclass
class Stopwatch:
    """
    一个很简单的秒表类，用来测代码跑了多久。

    用法：
      sw = Stopwatch()
      ...跑一段代码...
      print(sw.get_time())  # 看过去了多少秒

    或者：
      if sw.passed(5.0):
          print("已经过了5秒")
    """
    last_time: float = None  # 记录上一次“开始计时”的时间戳

    def __post_init__(self):
        # dataclass 创建对象后自动调用，用来做初始化
        self.restart()

    def restart(self):
        """重新开始计时（把当前时间当成起点）。"""
        self.last_time = time.time()

    def get_time(self) -> float:
        """返回从上次 restart() 到现在经过了多少秒。"""
        return time.time() - self.last_time

    def passed(self, duration: float) -> bool:
        """
        判断是否已经过去了 duration 秒。
        duration 单位是秒，比如 0.5 表示半秒。
        """
        return self.get_time() >= duration


def index(batched_struct):
    """
    让你可以用更舒服的方式从“批量 pytree”里取第 idx 个样本。

    batched_struct 的意思：
    - 它是一个 pytree（嵌套结构）
    - 但每个叶子（数组）第0维是 batch 维度
      例如 leaf.shape = (B, ...)

    你可能想做这种事：
      single = jax.tree_map(lambda x: x[idx], batched_struct)

    这个函数就是把上面的操作包成一个好用的语法糖，让你写：
      single = index(batched_struct)[idx]

    就像：
      batched_struct[idx]
    但 pytree 本身不支持直接这样索引，所以我们包一层类来实现。
    """
    class Indexer:
        def __getitem__(self, idx):
            # 对 pytree 的每个叶子 x，取 x[idx]
            # 返回的还是同样结构的 pytree，但去掉了 batch 维度
            return tree_util.tree_map(lambda x: x[idx], batched_struct)

    return Indexer()
