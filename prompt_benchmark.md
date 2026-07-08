阅读/Users/a1/Documents/workspace/ttnsOptimization/program_progress.md, 里面有完整的项目目标，模型说明，benchmark说明, 以及当前模型的调用方式.

你的任务: 在data中已经解压的数据集上测试全局chow-liu TTNS和全局TTDE的差异, 需要比较:1. 全局TTDE 2. 全局TTNS(不带平方归一化) 3. 全局TTNSDE(带平方归一化)

流程: 
1. 阅读代码， 确定使用的模型
2. 进行测试， 
3. 输出结果和图像， 其中图像需要包含切片密度图。
4. 对比结果并得出结论
5. 更新program_progress中的结果, 并且规划下一步任务
6. 无论结果如何提交commit

约束: 不要创建新的模型， 在已经实现的模型中找到对应所需要的模型，并确认其数学正确性.

