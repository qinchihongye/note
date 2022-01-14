

<center><font size='8' color='#6c549c' face=FangSong_GB2312>Sklearn</font></center>















  <center><font size="6" color='#6c549c' face=FangSong_GB2312>(开始时间：2019/10/21)<font></center>















<center><font size="6" color='#6c549c' face=FangSong_GB2312>(by 孟智超)<font></center>























# CATLOG

[toc]

# 一、决策树

## sklearn 中的决策树

### 关键概念、核心问题

* ==节点==
  1. 根节点：没有进边，有出边。包含最初的，针对特征的提问。 
  2. 中间节点：既有进边也有出边，进边只有一条，出边可以有很多条。都是针对特征的提问。
  3.  叶子节点：有进边，没有出边，每个叶子节点都是一个类别标签。 
  4. 子节点和父节点：在两个相连的节点中，更接近根节点的是父节点，另一个是子节点。

* 核心==问题==

  1. 如何从数据表中找出最佳节点和最佳分枝？ 

  2. 如何让决策树停止生长，防止过拟合？

### 模块sklearn.tree

*  `sklearn`中决策树的类都在”`tree`“这个模块之下。这个模块总共包含五个类：

  |             类              |                                       |
  | :-------------------------: | :-----------------------------------: |
  | tree.DecisionTreeClassifier |                分类树                 |
  | tree.DecisionTreeRegressor  |                回归树                 |
  |    tree.export_graphviz     | 将生成的决策树导出为DOT格式，画图专用 |
  |  tree.ExtraTreeClassifier   |          高随机版本的分类树           |
  |   tree.ExtraTreeRegressor   |          高随机版本的回归树           |

### sklearn的基本建模流程

* `sklearn`建模的基本流程

  ![image-20210820153802776](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021%2012%2023%20image-20210820153802776.png)

  在这个流程下，分类树对应的代码是：

  ```python
  from sklearn import tree                #导入需要的模块
  
  clf = tree.DecisionTreeClassifier()     #实例化
  clf = clf.fit(X_train,y_train)          #用训练集数据训练模型
  result = clf.score(X_test,y_test)       #导入测试集，从接口中调用需要的信息
  ```

### sklearn .metrics方法

* 获取`sklearn.metrics`中的所有评估方法

  ```python
  import sklearn
  
  sorted(sklearn.metrics.SCORERS.keys())
  
  """输出"""
  ['accuracy',
   'adjusted_mutual_info_score',
   'adjusted_rand_score',
   'average_precision',
   'completeness_score',
   'explained_variance',
   'f1',
   'f1_macro',
   'f1_micro',
   'f1_samples',
   'f1_weighted',
   'fowlkes_mallows_score',
   'homogeneity_score',
   'log_loss',
   'mean_absolute_error',
   'mean_squared_error',
   'median_absolute_error',
   'mutual_info_score',
   'neg_log_loss',
   'neg_mean_absolute_error',
   'neg_mean_squared_error',
   'neg_mean_squared_log_error',
   'neg_median_absolute_error',
   'normalized_mutual_info_score',
   'precision',
   'precision_macro',
   'precision_micro',
   'precision_samples',
   'precision_weighted',
   'r2',
   'recall',
   'recall_macro',
   'recall_micro',
   'recall_samples',
   'recall_weighted',
   'roc_auc',
   'v_measure_score']
  ```

  

---

## 分类树

* `sklearn.tree.DecisionTreeClassifier`

  ```python
  sklearn.tree.DecisionTreeClassifier (criterion=’gini’ # 不纯度计算方法
                                       , splitter=’best’ # best & random
                                       , max_depth=None # 树最大深度
                                       , min_samples_split=2 # 当前节点可划分最少样本数
                                       , min_samples_leaf=1 # 子节点最少样本数
                                       , min_weight_fraction_leaf=0.0
                                       , max_features=None
                                       , random_state=None
                                       , max_leaf_nodes=None
                                       , min_impurity_decrease=0.0
                                       , min_impurity_split=None
                                       , class_weight=None
                                       , presort=False
                                      )
  ```

### 重要参数

#### criterion

* `criterion`这个参数正是用来决定不纯度的计算方法的。

  `sklearn`提供了两种选择：

  1. 输入”==entropy==“，使用信息熵（`Entropy`），`sklearn`实际计算的是基于信息熵的信息增益(==Information Gain==)，即父节点的信息熵和子节点的信息熵之差。 
  2.  输入”==gini==“，使用基尼系数（==Gini Impurity==）

  $$
  Entropy(t) = - \sum \limits_{i=0}^{c-1} p(i|t)\log{_2}p(i|t)
  $$

  $$
  Gini(t) = 1 - \sum_{i=0}^{c-1}p(i|t)^2
  $$

  其中t代表给定的节点，i代表标签的任意分类，$p(i|t)$ 代表标签分类i在节点t上所占的比例。注意，当使用信息熵 时，`sklearn`实际计算的是基于信息熵的信息增益(==Information Gain==)，即父节点的信息熵和子节点的信息熵之差。 比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。但是在实际使用中，信息熵和基尼系数的效果基 本相同。信息熵的计算比基尼系数缓慢一些，因为基尼系数的计算不涉及对数。另外，因为信息熵对不纯度更加敏 感，所以信息熵作为指标时，决策树的生长会更加“精细”，因此对于高维数据或者噪音很多的数据，信息熵很容易 过拟合，基尼系数在这种情况下效果往往比较好。当模型拟合程度不足的时候，即当模型在训练集和测试集上都表 现不太好的时候，使用信息熵。当然，这些不是绝对的。

  | 参数                |                          criterion                           |
  | ------------------- | :----------------------------------------------------------: |
  | 如何影响模型?       | 确定不纯度的计算方法，帮忙找出最佳节点和最佳分枝，不纯度越低，决策树对训练集 的拟合越好 |
  | 可能的输入有哪 些？ | 不填默认基尼系数，填写gini使用基尼系数，填写entropy使用信息增益 |
  | 怎样选取参数？      | 通常就使用基尼系数 数据维度很大，噪音很大时使用基尼系数 维度低，数据比较清晰的时候，信息熵和基尼系数没区别 当决策树的拟合程度不够的时候，使用信息熵，两个都试试不好就换另一个 |

  ```python
  # -*- coding: utf-8 -*-
  
  """
  **************************************************
  @author:   Ying                                      
  @software: PyCharm                       
  @file: 分类树_criterion.py
  @time: 2021-08-20 16:13                          
  **************************************************
  """
  from sklearn import tree
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import graphviz
  
  # 加载数据
  wine = load_wine()
  data = pd.DataFrame(wine.data, columns=wine.feature_names)  # X
  target = pd.DataFrame(wine.target)  # y
  
  # 划分训练测试集
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
  
  # 两种criterion
  
  for criterion_ in ['entropy', 'gini']:
      clf = tree.DecisionTreeClassifier(criterion=criterion_)
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)  # 返回预测的准确度
      print(f'criterion:{criterion_} \t accurancy : {score}')
  
      # 保存决策树图
      feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类',
                      '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
  
      dot_data = tree.export_graphviz(clf
                                      , feature_names=feature_name
                                      , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                      , filled=True  # 填充颜色
                                      , rounded=True  # 圆角
                                      )
      graph = graphviz.Source(dot_data)
  
      graph.render(view=True, format="pdf", filename=f"decisiontree_pdf_{criterion_}")
  
      # 特征重要性
      feature_importances = clf.feature_importances_
      for i in [*zip(feature_name, feature_importances)]:
          print(i)
      print()
  ```
  
  ​	![image-20210820164533475](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021%2012%2023%20image-20210820164533475.png)
  
  ```python
  """输出如下"""
  
  criterion:entropy 	 accurancy : 0.8703703703703703
  ('酒精', 0.0)
  ('苹果酸', 0.0)
  ('灰', 0.0)
  ('灰的碱性', 0.02494246008989065)
  ('镁', 0.0)
  ('总酚', 0.0)
  ('类黄酮', 0.3296114164674079)
  ('非黄烷类酚类', 0.0)
  ('花青素', 0.0)
  ('颜色强度', 0.14329965511242485)
  ('色调', 0.0)
  ('od280/od315稀释葡萄酒', 0.0)
  ('脯氨酸', 0.5021464683302767)
  
  criterion:gini 	 accurancy : 0.8148148148148148
  ('酒精', 0.0)
  ('苹果酸', 0.0)
  ('灰', 0.0)
  ('灰的碱性', 0.0)
  ('镁', 0.04779989924874613)
  ('总酚', 0.06725255711062922)
  ('类黄酮', 0.3230308396876504)
  ('非黄烷类酚类', 0.0)
  ('花青素', 0.0235378291755189)
  ('颜色强度', 0.0)
  ('色调', 0.0)
  ('od280/od315稀释葡萄酒', 0.0878400745934749)
  ('脯氨酸', 0.45053880018398057)
  ```

* 回到模型步骤，每次运行==score==会在某个值附近 波动，引起每次画出来的每一棵树都不一样。它为什么会不稳定呢？如果使用其他数据集，它还会不稳定吗？

  无论决策树模型如何进化，在分枝上的本质都还是追求某个不纯度相关的指标的优化，而正如我 们提到的，不纯度是基于节点来计算的，也就是说，决策树在建树时，是靠优化节点来追求一棵优化的树，但最优 的节点能够保证最优的树吗？集成算法被用来解决这个问题：sklearn表示，既然一棵树不能保证最优，那就建更 多的不同的树，然后从中取最好的。怎样从一组数据集中建不同的树？在每次分枝时，不从使用全部特征，而是随 机选取一部分特征，从中选取不纯度相关指标最优的作为分枝用的节点。这样，每次生成的树也就不同了。

#### random_state&spliter

* `random_state`用来设置分枝中的随机模式的参数，默认`None`，在高维度时随机性会表现更明显，低维度的数据 （比如鸢尾花数据集），随机性几乎不会显现。输入任意整数，会一直长出同一棵树，让模型稳定下来。

*  `splitter`也是用来控制决策树中的随机选项的，有两种输入值:

  1. best
  2. random

  输入”==best=="，决策树在分枝时虽然随机，但是还是会 优先选择更重要的特征进行分枝（重要性可以通过属性`feature_importances_`查看）

  输入“==random=="，决策树在 分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这 也是防止过拟合的一种方式。当你预测到你的模型会过拟合，用这两个参数来帮助你降低树建成之后过拟合的可能 性。当然，树一旦建成，我们依然是使用剪枝参数来防止过拟合。

  ```python
  # -*- coding: utf-8 -*-
  
  """
  **************************************************
  @author:   Ying                                      
  @software: PyCharm                       
  @file: 2、分类树_random_state&spliter.py
  @time: 2021-08-20 16:58                          
  **************************************************
  """
  
  from sklearn import tree
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import graphviz
  
  # 加载数据
  wine = load_wine()
  data = pd.DataFrame(wine.data, columns=wine.feature_names)  # X
  target = pd.DataFrame(wine.target)  # y
  
  # 划分训练测试集
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
  
  clf = tree.DecisionTreeClassifier(criterion='gini'
                                    , random_state=30
                                    , splitter='best')
  
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)  # 返回预测的准确度
  
  # 保存决策树图
  feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类',
                  '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
  
  dot_data = tree.export_graphviz(clf
                                  , feature_names=feature_name
                                  , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                  , filled=True  # 填充颜色
                                  , rounded=True  # 圆角
                                  )
  graph = graphviz.Source(dot_data)
  
  graph.render(view=True, format="pdf", filename="decisiontree_pdf")
  
  # 特征重要性
  feature_importances = clf.feature_importances_
  
  a = pd.DataFrame([*zip(feature_name, feature_importances)])
  a.columns = ['feature', 'importance']
  a.sort_values('importance', ascending=False, inplace=True)
  print(a)
  
  ```

### 剪枝参数

* 在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止。这样的决策树 往往会过拟合。为了让决策树有更好的泛化性，我们要对决策树进行剪枝。剪枝策略对决策树的影响巨大，正确的剪枝策略是优化 决策树算法的核心。`sklearn`为我们提供了不同的剪枝策略：

#### max_depth

* 限制树的最大深度，超过设定深度的树枝全部剪掉

  这是用得最广泛的剪枝参数，在高维度低样本量时非常有效。决策树多生长一层，对样本量的需求会增加一倍，所 以限制树深度能够有效地限制过拟合。在集成算法中也非常实用。实际使用时，建议从=3开始尝试，看看拟合的效 果再决定是否增加设定深度。

#### min_samples_leaf & min_samples_split

* `min_samples_leaf`限定，一个节点在分枝后的每个子节点都必须包含至少`min_samples_leaf`个训练样本，否则分 枝就不会发生，或者，分枝会朝着满足每个子节点都包含`min_samples_leaf`个样本的方向去发生。

  `min_samples_leaf`限定，一个节点在分枝后的每个子节点都必须包含至少`min_samples_leaf`个训练样本，否则分 枝就不会发生，或者，分枝会朝着满足每个子节点都包含`min_samples_leaf`个样本的方向去发生 一般搭配`max_depth`使用，在回归树中有神奇的效果，可以让模型变得更加平滑。这个参数的数量设置得太小会引 起过拟合，设置得太大就会阻止模型学习数据。一般来说，建议从===5==开始使用。如果叶节点中含有的样本量变化很 大，建议输入浮点数作为样本量的百分比来使用。同时，这个参数可以保证每个叶子的最小尺寸，可以在回归问题 中避免低方差，过拟合的叶子节点出现。对于类别不多的分类问题，===1==通常就是最佳选择。 

* `min_samples_split`限定，一个节点必须要包含至少`min_samples_split`个训练样本，这个节点才允许被分枝，否则 分枝就不会发生。

* `min_samples_leaf` = 10

  `min_samples_split` = 25

  ```python
  # -*- coding: utf-8 -*-
  
  """
  **************************************************
  @author:   Ying                                      
  @software: PyCharm                       
  @file: 3、分类树_min_samples_leaf& min_samples_split.py
  @time: 2021-08-26 10:51                          
  **************************************************
  """
  from sklearn import tree
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import graphviz
  
  # 加载数据
  wine = load_wine()
  data = pd.DataFrame(wine.data, columns=wine.feature_names)  # X
  target = pd.DataFrame(wine.target)  # y
  
  # 划分训练测试集
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
  
  clf = tree.DecisionTreeClassifier(criterion="entropy"
                                    , random_state=30
                                    , splitter="random"
                                    , max_depth=3
                                    , min_samples_leaf=10
                                    , min_samples_split=25
                                    )
  
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)  # 返回预测的准确度
  
  # 保存决策树图
  feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类',
                  '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
  
  dot_data = tree.export_graphviz(clf
                                  , feature_names=feature_name
                                  , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                  , filled=True  # 填充颜色
                                  , rounded=True  # 圆角
                                  )
  graph = graphviz.Source(dot_data)
  
  graph.render(view=True, format="png", filename="./save/decisiontree_pdf")
  
  # 特征重要性
  feature_importances = clf.feature_importances_
  
  a = pd.DataFrame([*zip(feature_name, feature_importances)])
  a.columns = ['feature', 'importance']
  a.sort_values('importance', ascending=False, inplace=True)
  print(a)
  ```

  ​	![image-20210826110623238](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021%2008%2026%20image-20210826110623238.png)

#### max_features & min_impurity_decrease

* 一般max_depth使用，用作树的”精修“ max_features限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。和max_depth异曲同工， max_features是用来限制高维度数据的过拟合的剪枝参数，但其方法比较暴力，是直接限制可以使用的特征数量 而强行使决策树停下的参数，在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型 学习不足。如果希望通过降维的方式防止过拟合，建议使用PCA，ICA或者特征选择模块中的降维算法。 min_impurity_decrease限制信息增益的大小，信息增益小于设定数值的分枝不会发生。这是在0.19版本中更新的 功能，在0.19版本之前时使用min_impurity_split。

#### 确定最优剪枝参数(超参数曲线)

* 那具体怎么来确定每个参数填写什么值呢？这时候，我们就要使用确定超参数的曲线来进行判断了，继续使用我们 已经训练好的决策树模型clf。超参数的学习曲线，是一条以超参数的取值为横坐标，模型的度量指标为纵坐标的曲 线，它是用来衡量不同超参数取值下模型的表现的线。在我们建好的决策树里，我们的模型度量指标就是score。

  ```python
  # -*- coding: utf-8 -*-
  
  """
  **************************************************
  @author:   Ying                                      
  @software: PyCharm                       
  @file: 4、分类树_超参数曲线.py
  @time: 2021-12-01 11:28                          
  **************************************************
  """
  
  from sklearn import tree
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import matplotlib.pyplot as plt
  
  # 加载数据
  wine = load_wine()
  data = pd.DataFrame(wine.data, columns=wine.feature_names)  # X
  target = pd.DataFrame(wine.target)  # y
  
  # 划分训练测试集
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
  
  
  
  test = []
  for i in range(10):
      clf = tree.DecisionTreeClassifier(max_depth=i+1
                                        ,criterion="entropy"
                                        ,random_state=10
                                        ,splitter='random')
      clf = clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      test.append(score)
  plt.plot(range(1,11),test,color="red",label="max_depth")
  plt.legend()
  plt.show()
  ```

  ![image-20211201113350789](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021%2012%2001%20image-20211201113350789.png)

---

### 目标权重参数

* class_weight & min_weight_fraction_leaf
  完成样本标签平衡的参数。样本不平衡是指在一组数据集中，标签的一类天生占有很大的比例。比如说，在银行要 判断“一个办了信用卡的人是否会违约”，就是是vs否(1%:99%)的比例。这种分类状况下，即便模型什么也不 做，全把结果预测成“否”，正确率也能有99%。因此我们要使用class_weight参数对样本标签进行一定的均衡，给 少量的标签更多的权重，让模型更偏向少数类，向捕获少数类的方向建模。该参数默认None，此模式表示自动给 与数据集中的所有标签相同的权重。
  有了权重之后，样本量就不再是单纯地记录数目，而是受输入的权重影响了，因此这时候剪枝，就需要搭配min_ weight_fraction_leaf这个基于权重的剪枝参数来使用。另请注意，基于权重的剪枝参数(例如min_weight_ fraction_leaf)将比不知道样本权重的标准(比如min_samples_leaf)更少偏向主导类。如果样本是加权的，则使 用基于权重的预修剪标准来更容易优化树结构，这确保叶节点至少包含样本权重的总和的一小部分。

### 重要属性和接口 

* 属性是在模型训练之后，能够调用查看的模型的各种性质。对决策树来说，最重要的是`feature_importances_`，能够查看各个特征对模型的重要性。
  sklearn中许多算法的接口都是相似的，比如说我们之前已经用到的fit和score，几乎对每个算法都可以使用。除了 这两个接口之外，决策树最常用的接口还有`apply`和`predict`。apply中输入测试集返回每个测试样本所在的叶子节 点的索引，`predict`输入测试集返回每个测试样本的标签。

* 所有接口中要求输入X_train和X_test的部分，输入的特征矩阵必须至少是一个二维矩阵。 sklearn不接受任何一维矩阵作为特征矩阵被输入。如果你的数据的确只有一个特征，那必须用reshape(-1,1)来给 矩阵增维;如果你的数据只有一个特征和一个样本，使用reshape(1,-1)来给你的数据增维。

  ```python
  #apply返回每个测试样本所在的叶子节点的索引 
  clf.apply(Xtest)
  #predict返回每个测试样本的分类/回归结果 
  clf.predict(Xtest)
  ```

### 参数总结

* 分类树的`八个参数，一个属性，四个接口`，。
  1. 八个参数:Criterion，两个随机性相关的参数(random_state，splitter)，五个剪枝参数(max_depth, min_samples_split，min_samples_leaf，max_feature，min_impurity_decrease)
  2. 一个属性:feature_importances_

四个接口:fit，score，apply，predict 

---

## 回归树

* `class sklearn.tree.DecisionTreeRegressor`

  ```python
  sklearn.tree.DecisionTreeRegressor (criterion=’mse’ # 回归树不纯度计算方法(均方误差)
                                     , splitter=’best’ # best & random
                                     , max_depth=None # 树最大深度
                                     , min_samples_split=2 # 当前节点可划分最少样本数
                                     , min_samples_leaf=1 # 子节点最少样本数
                                     , min_weight_fraction_leaf=0.0
                                     , max_features=None # 最大特征数（默认为特征数开平方后取整）
                                     , random_state=None # 随机数种子
                                     , max_leaf_nodes=None 
                                     , min_impurity_decrease=0.0
                                     , min_impurity_split=None
                                     , presort=False
                                  )
  ```

  几乎所有参数，属性及接口都和分类树一模一样。需要注意的是，在回归树种，没有标签分布是否均衡的问题，因 此没有class_weight这样的参数。

### 重要参数、属性、接口

* 参数`criterion`
  回归树衡量分枝质量的指标，支持的标准有三种:

  1. 输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，这种方法通过使用叶子节点的均值来最小化L2损失 

  2. 输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差

  3. 输入"mae"使用绝对平均误差MAE(mean absolute error)，这种指标使用叶节点的中值来最小化L1损失 
     $$
     M S E=\frac{1}{N} \sum_{i=1}^{N}\left(f_{i}-y_{i}\right)^{2}
     $$

  其中N是样本数量，i是每一个数据样本，fi是模型回归出的数值，yi是样本点i实际的数值标签。所以MSE的本质， 其实是样本真实数据与回归结果的差异。在回归树中，MSE不只是我们的分枝质量衡量指标，也是我们最常用的衡 量回归树回归质量的指标，当我们在使用交叉验证，或者其他方式获取回归树的结果时，我们往往选择均方误差作 为我们的评估(在分类树中这个指标是score代表的预测准确率)。在回归中，我们追求的是，MSE越小越好。

  然而，回归树的接口score返回的是R平方，并不是MSE。R平方被定义如下:
  $$
  \begin{array}{c}
  R^{2}=1-\frac{u}{v} \\
  u=\sum_{i=1}^{N}\left(f_{i}-y_{i}\right)^{2} \quad v=\sum_{i=1}^{N}\left(y_{i}-\hat{y}\right)^{2}
  \end{array}
  $$
  

  其中u是残差平方和(MSE * N)，v是总平方和，N是样本数量，i是每一个数据样本，fi是模型回归出的数值，yi 是样本点i实际的数值标签。y帽是真实数值标签的平均数。R平方可以为正为负(如果模型的残差平方和远远大于 模型的总平方和，模型非常糟糕，R平方就会为负)，而均方误差永远为正。

  虽然均方误差永远为正，但是sklearn当中使用均方误差作为评判标准时，却是计算”负均差“(neg_mean_squared_error)。这是因为sklearn在计算模型评估指标的时候，会考虑指标本身的性质，均 方误差本身是一种误差，所以被sklearn划分为模型的一种损失(loss)，因此在sklearn当中，都以负数表示。真正的 均方误差MSE的数值，其实就是neg_mean_squared_error去掉负号的数字。

* 属性中最重要的依然是feature_importances_，接口依然是apply, fit, predict, score最核心。

### 交叉验证

* 回归树_交叉验证

  ```python
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_boston
  from sklearn.model_selection import cross_val_score
  
  boston = load_boston()
  regressor = DecisionTreeRegressor(random_state=10)
  cross_val_score(regressor # 模型
                  ,boston.data # X
                  ,boston.target # y
                  ,cv=10 # 交叉验证次数
                  ,verbose=False # 训练过程是否打印
                  ,scoring='neg_mean_squared_error' # 负均方误差
                 ).mean()
  ```

  交叉验证是用来观察模型的稳定性的一种方法，我们将数据划分为n份，依次使用其中一份作为测试集，其他n-1份 作为训练集，多次计算模型的精确性来评估模型的平均准确程度。训练集和测试集的划分会干扰模型的结果，因此 用交叉验证n次的结果求出的平均值，是对模型效果的一个更好的度量。

  ​	![20211202Xze8q4](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 02 Xze8q4.png)

## 实例

### 回归树实例：正弦函数（带噪声）决策树回归

* 代码

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  from sklearn.tree import DecisionTreeRegressor
  
  %matplotlib inline
  
  # 创建一条含有噪声的正弦曲线
  rng = np.random.RandomState(1)
  X = np.sort(5*rng.rand(80,1),axis=0)
  y = np.sin(x).ravel()
  y[::5] += 3*(0.5-rng.rand(16))
  
  # 实例化训练模型
  regr_1 = DecisionTreeRegressor(max_depth=2)
  regr_2 = DecisionTreeRegressor(max_depth=5)
  regr_1.fit(X, y)
  regr_2.fit(X, y)
  
  # 测试集导入模型，测试结果
  X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
  y_1 = regr_1.predict(X_test)
  y_2 = regr_2.predict(X_test)
  
  # 绘制图像
  plt.figure(figsize=[12,8])
  plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
  plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
  plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
  plt.xlabel("data")
  plt.ylabel("target")
  plt.title("Decision Tree Regression")
  plt.legend()
  plt.show()
  ```

  ​	![20211202Akp8Ko](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 02 Akp8Ko.jpg)

  决策树学习了近似正弦曲线的局部回归，可以看到，若树的最大深度设置太高，则决策树学的太精细，它从训练数据中学到了很多细节，从而使模型偏离真实的正弦曲线，形成过拟合。

---

### 分类树实例：分类树在合成数据集上的表现

#### 代码分解

* 在不同结构的据集上测试一下决策树的效果（二分型，月亮形，环形）

* 导入

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  from matplotlib.colors import ListedColormap
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.datasets import make_moons,make_circles,make_classification
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import cross_val_score
  ```



* 生成三种数据集（月亮形、环形、二分型）

  ```
  X,y = make_classification(n_samples=100 # 生成100个样本
                            ,n_features=2 # 包含两个特征
                            ,n_redundant=0 # 添加冗余特征0个
                            ,n_informative=2 # 包含信息的特征是2个
                            ,random_state=1 # 随机数种子
                            ,n_clusters_per_class=1 # 每个簇内包含的标签类别有2个
                           )
  plt.scatter(X[:,0],X[:,1])
  ```

  ​	![20211206Sw3qqx](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 06 Sw3qqx.jpg)

  图上可以看出，生成的二分型数据的两个簇离彼此很远，可能分类边界过于明显，使用交叉验证尝试看一下是否分类效果特别好

  ```python
  for i in range(1,11):
      clf = DecisionTreeClassifier()
      cross_val = cross_val_score(clf,X,y,cv=10).mean()
      print(cross_val)
      
  """输出"""
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  ```

  交叉验证看，效果太好，这样不利于我们测试分类器的效果，因此我们使用np生成 随机数组通过让已经生成的二分型数据点加减0~1之间的随机数使数据分布变得更散更稀疏 注意，这个过程只能够运行一次，因为多次运行之后X会变得非常稀疏，两个簇的数据会混合在一起，分类器的效应会 继续下降。
  
  ```python
  rng = np.random.RandomState(2) # 生成一种随机模式
  X += 2*rng.uniform(size=X.shape)# 加减0-1之间的随机数
  Linearly_separable = (X,y)
  plt.figure(figsize=(12,8))
  plt.scatter(X[:,0],X[:,1])
  ```
  
  ​	![202112065QEFAm](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 06 5QEFAm.jpg)
  
  依次生成月亮型环形数据，并并入datasets中
  
  ```python
  # 用make_moons创建月亮型数据
  X,y = make_moons(noise=0.3
                   ,random_state=0
                  )
  Moons_data = (X,y)
  
  # 用make_circles创建环形数据
  X,y = make_circles(noise=0.2
                     ,factor=0.5
                     ,random_state=1
                    )
  Circle_data = (X,y)
  
  datasets = (Moons_data,Circle_data,Linearly_separable)
  ```



* 画出三种数据集和三颗决策树的分类效应图像

  ```python
  """画出三种数据集和三颗决策树的分类效应图像"""
  figure = plt.figure(figsize=(12,18))
  i = 1 # 设置用来安排图像显示位置的全局变量i
  
  # 开始迭代数据，对datasets中的数据进行for循环
  
  for ds_index,ds in enumerate(datasets):
      X,y = ds
      X = StandardScaler().fit_transform(X)
      X_train,X_test,y_train,y_test = train_test_split(X
                                                       ,y
                                                       ,test_size=.4
                                                       ,random_state=42)
      # 找出数据集中两个特征的最大值和最小值，让最大值+0.5，最小值-0.5，创造一个比两个特征的区间本身更大 一点的区间
      x1_min,x1_max = X[:,0].min() - .5,X[:,0].max() + .5
      x2_min,x2_max = X[:,1].min() - .5,X[:,1].max() + .5
      
      #用特征向量生成网格数据，网格数据，其实就相当于坐标轴上无数个点 
      #函数np.arange在给定的两个数之间返回均匀间隔的值，0.2为步长 
      #函数meshgrid用以生成网格数据，能够将两个一维数组生成两个二维矩阵。 
      #如果第一个数组是narray，维度是n，第二个参数是marray，维度是m。那么生成的第一个二维数组是以narray为行，m行的矩阵，而第二个二维数组是以marray的转置为列，n列的矩阵 
      #生成的网格数据，是用来绘制决策边界的，因为绘制决策边界的函数contourf要求输入的两个特征都必须是二维的
      array1,array2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2),
                           np.arange(x2_min, x2_max, 0.2))
      #接下来生成彩色画布 
      #用ListedColormap为画布创建颜色，#FF0000正红，#0000FF正蓝 
      cm = plt.cm.RdBu
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      #在画布上加上一个子图，数据为len(datasets)行，2列，放在位置i上 
      ax = plt.subplot(len(datasets), 2, i)
      #到这里为止，已经生成了0~1之间的坐标系3个了，接下来为我们的坐标系放上标题 
      #我们有三个坐标系，但我们只需要在第一个坐标系上有标题，因此设定if ds_index==0这个条件 
      if ds_index == 0:
          ax.set_title("Input data")
      #将数据集的分布放到我们的坐标系上
      #先放训练集
      ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                 cmap=cm_bright,edgecolors='k')
      # 放测试集
      ax.scatter(X_test[:, 0]
                 , X_test[:,1]
                 ,c=y_test
                 ,cmap=cm_bright
                 ,alpha=0.6
                 ,edgecolors='k'
                )
      #为图设置坐标轴的最大值和最小值，并设定没有坐标轴 
      ax.set_xlim(array1.min(), array1.max()) 
      ax.set_ylim(array2.min(), array2.max()) 
      ax.set_xticks(())
      ax.set_yticks(())
      
      #每次循环之后，改变i的取值让图每次位列不同的位置
      i+=1
      
      #至此为止，数据集本身的图像已经布置完毕，运行以上的代码，可以看见三个已经处理好的数据集 
      #############################从这里开始是决策树模型##########################
      #在这里，len(datasets)其实就是3，2是两列 
      #在函数最开始，我们定义了i=1，并且在上边建立数据集的图像的时候，已经让i+1,所以i在每次循环中的取值是2，4，6
      ax = plt.subplot(len(datasets),2,i)
      #决策树的建模过程:实例化 → fit训练 → score接口得到预测的准确率 
      clf = DecisionTreeClassifier(max_depth=5) 
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      #绘制决策边界，为此，我们将为网格中的每个点指定一种颜色[x1_min，x1_max] x [x2_min，x2_max] 
      #分类树的接口，predict_proba，返回每一个输入的数据点所对应的标签类概率 
      #类概率是数据点所在的叶节点中相同类的样本数量/叶节点中的样本总数量 
      #由于决策树在训练的时候导入的训练集X_train里面包含两个特征，所以我们在计算类概率的时候，也必须导入结构相同的数组，即是说，必须有两个特征 
      #ravel()能够将一个多维数组转换成一维数组 
      #np.c_是能够将两个数组组合起来的函数
      
      #在这里，我们先将两个网格数据降维降维成一维数组，再将两个数组链接变成含有两个特征的数据，再带入决策 树模型，生成的Z包含数据的索引和每个样本点对应的类概率，再切片，切出类概率
      Z = clf.predict_proba(np.c_[array1.ravel(),array2.ravel()])[:, 1]
      #np.c_[np.array([1,2,3]), np.array([4,5,6])]
      #将返回的类概率作为数据，放到contourf里面绘制去绘制轮廓
      Z = Z.reshape(array1.shape)
      ax.contourf(array1, array2, Z, cmap=cm, alpha=.8)
      #将数据集的分布放到我们的坐标系上
      # 将训练集放到图中去
      ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                  edgecolors='k') 
      # 将测试集放到图中去
      ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                 edgecolors='k', alpha=0.6)
      #为图设置坐标轴的最大值和最小值 
      ax.set_xlim(array1.min(), array1.max()) 
      ax.set_ylim(array2.min(), array2.max()) 
      #设定坐标轴不显示标尺也不显示数字 
      ax.set_xticks(())
      ax.set_yticks(())
      #我们有三个坐标系，但我们只需要在第一个坐标系上有标题，因此设定if ds_index==0这个条件 
      if ds_index == 0:
          ax.set_title("Decision Tree")
      #写在右下角的数字
      ax.text(array1.max() - .3, array2.min() + .3, ('{:.1f}%'.format(score*100)),
              size=15, horizontalalignment='right') 
      
      #让i继续加一
      i += 1
      
  plt.tight_layout()
  plt.show()
      
  ```

  ​	![20211206CLVzAn](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 06 CLVzAn.jpg)

* 从图上来看，每一条线都是决策树在二维平面上画出的一条决策边界，每当决策树分枝一次，就有一条线出现。当数据的维度更高的时候，这条决策边界就会由线变成面，甚至变成我们想象不出的多维图形。

  同时，很容易看得出，**分类树天生不擅长环形数据**。每个模型都有自己的决策上限，所以一个怎样调整都无法提升 表现的可能性也是有的。当一个模型怎么调整都不行的时候，我们可以选择换其他的模型使用，不要在一棵树上吊 死。顺便一说:

  1. 最擅长月亮型数据的是最近邻算法，RBF支持向量机和高斯过程;
  2. 最擅长环形数据的是最近邻算法 和高斯过程;
  3.  最擅长对半分的数据的是朴素贝叶斯，神经网络和随机森林。

#### 所有代码

* 所有代码

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  from matplotlib.colors import ListedColormap
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.datasets import make_moons,make_circles,make_classification
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import cross_val_score
  
  """生成三种数据集（月亮形、环形、二分型）"""
  #make_classification生成二分型数据
  X,y = make_classification(n_samples=100 # 生成100个样本
                            ,n_features=2 # 包含两个特征
                            ,n_redundant=0 # 添加冗余特征0个
                            ,n_informative=2 # 包含信息的特征是2个
                            ,random_state=1 # 随机数种子
                            ,n_clusters_per_class=1 # 每个簇内包含的标签类别有2个
                           )
  
  for i in range(1,11):
      clf = DecisionTreeClassifier()
      cross_val = cross_val_score(clf,X,y,cv=10).mean()
      print(cross_val)
  plt.figure(figsize=(12,8))
  plt.scatter(X[:,0],X[:,1])
  plt.title('make_classification')
  plt.show()
  
  ##从交叉验证和图上可以看出，生成的二分型数据的两个簇离彼此很远
  # ，这样不利于我们测试分类器的效果，因此我们使用np生成 随机数组
  # ，通过让已经生成的二分型数据点加减0~1之间的随机数
  # ，使数据分布变得更散更稀疏 
  #注意，这个过程只能够运行一次，因为多次运行之后X会变得非常稀疏，两个簇的数据会混合在一起，分类器的效应会 继续下降
  rng = np.random.RandomState(2) # 生成一种随机模式
  X += 2*rng.uniform(size=X.shape)# 加减0-1之间的随机数
  Linearly_separable = (X,y)
  plt.figure(figsize=(12,8))
  plt.scatter(X[:,0],X[:,1])
  
  
  
  # 用make_moons创建月亮型数据
  X,y = make_moons(noise=0.3
                   ,random_state=0
                  )
  Moons_data = (X,y)
  
  # 用make_circles创建环形数据
  X,y = make_circles(noise=0.2
                     ,factor=0.5
                     ,random_state=1
                    )
  Circle_data = (X,y)
  
  datasets = (Moons_data,Circle_data,Linearly_separable)
  
  """画出三种数据集和三颗决策树的分类效应图像"""
  figure = plt.figure(figsize=(12,18))
  i = 1 # 设置用来安排图像显示位置的全局变量i
  
  # 开始迭代数据，对datasets中的数据进行for循环
  
  for ds_index,ds in enumerate(datasets):
      X,y = ds
      X = StandardScaler().fit_transform(X)
      X_train,X_test,y_train,y_test = train_test_split(X
                                                       ,y
                                                       ,test_size=.4
                                                       ,random_state=42)
      # 找出数据集中两个特征的最大值和最小值，让最大值+0.5，最小值-0.5，创造一个比两个特征的区间本身更大 一点的区间
      x1_min,x1_max = X[:,0].min() - .5,X[:,0].max() + .5
      x2_min,x2_max = X[:,1].min() - .5,X[:,1].max() + .5
      
      #用特征向量生成网格数据，网格数据，其实就相当于坐标轴上无数个点 
      #函数np.arange在给定的两个数之间返回均匀间隔的值，0.2为步长 
      #函数meshgrid用以生成网格数据，能够将两个一维数组生成两个二维矩阵。 
      #如果第一个数组是narray，维度是n，第二个参数是marray，维度是m。那么生成的第一个二维数组是以narray为行，m行的矩阵，而第二个二维数组是以marray的转置为列，n列的矩阵 
      #生成的网格数据，是用来绘制决策边界的，因为绘制决策边界的函数contourf要求输入的两个特征都必须是二维的
      array1,array2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2),
                           np.arange(x2_min, x2_max, 0.2))
      #接下来生成彩色画布 
      #用ListedColormap为画布创建颜色，#FF0000正红，#0000FF正蓝 
      cm = plt.cm.RdBu
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      #在画布上加上一个子图，数据为len(datasets)行，2列，放在位置i上 
      ax = plt.subplot(len(datasets), 2, i)
      #到这里为止，已经生成了0~1之间的坐标系3个了，接下来为我们的坐标系放上标题 
      #我们有三个坐标系，但我们只需要在第一个坐标系上有标题，因此设定if ds_index==0这个条件 
      if ds_index == 0:
          ax.set_title("Input data")
      #将数据集的分布放到我们的坐标系上
      #先放训练集
      ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                 cmap=cm_bright,edgecolors='k')
      # 放测试集
      ax.scatter(X_test[:, 0]
                 , X_test[:,1]
                 ,c=y_test
                 ,cmap=cm_bright
                 ,alpha=0.6
                 ,edgecolors='k'
                )
      #为图设置坐标轴的最大值和最小值，并设定没有坐标轴 
      ax.set_xlim(array1.min(), array1.max()) 
      ax.set_ylim(array2.min(), array2.max()) 
      ax.set_xticks(())
      ax.set_yticks(())
      
      #每次循环之后，改变i的取值让图每次位列不同的位置
      i+=1
      
      #至此为止，数据集本身的图像已经布置完毕，运行以上的代码，可以看见三个已经处理好的数据集 
      #############################从这里开始是决策树模型##########################
      #在这里，len(datasets)其实就是3，2是两列 
      #在函数最开始，我们定义了i=1，并且在上边建立数据集的图像的时候，已经让i+1,所以i在每次循环中的取值是2，4，6
      ax = plt.subplot(len(datasets),2,i)
      #决策树的建模过程:实例化 → fit训练 → score接口得到预测的准确率 
      clf = DecisionTreeClassifier(max_depth=5) 
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      #绘制决策边界，为此，我们将为网格中的每个点指定一种颜色[x1_min，x1_max] x [x2_min，x2_max] 
      #分类树的接口，predict_proba，返回每一个输入的数据点所对应的标签类概率 
      #类概率是数据点所在的叶节点中相同类的样本数量/叶节点中的样本总数量 
      #由于决策树在训练的时候导入的训练集X_train里面包含两个特征，所以我们在计算类概率的时候，也必须导入结构相同的数组，即是说，必须有两个特征 
      #ravel()能够将一个多维数组转换成一维数组 
      #np.c_是能够将两个数组组合起来的函数
      
      #在这里，我们先将两个网格数据降维降维成一维数组，再将两个数组链接变成含有两个特征的数据，再带入决策 树模型，生成的Z包含数据的索引和每个样本点对应的类概率，再切片，切出类概率
      Z = clf.predict_proba(np.c_[array1.ravel(),array2.ravel()])[:, 1]
      #np.c_[np.array([1,2,3]), np.array([4,5,6])]
      #将返回的类概率作为数据，放到contourf里面绘制去绘制轮廓
      Z = Z.reshape(array1.shape)
      ax.contourf(array1, array2, Z, cmap=cm, alpha=.8)
      #将数据集的分布放到我们的坐标系上
      # 将训练集放到图中去
      ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                  edgecolors='k') 
      # 将测试集放到图中去
      ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                 edgecolors='k', alpha=0.6)
      #为图设置坐标轴的最大值和最小值 
      ax.set_xlim(array1.min(), array1.max()) 
      ax.set_ylim(array2.min(), array2.max()) 
      #设定坐标轴不显示标尺也不显示数字 
      ax.set_xticks(())
      ax.set_yticks(())
      #我们有三个坐标系，但我们只需要在第一个坐标系上有标题，因此设定if ds_index==0这个条件 
      if ds_index == 0:
          ax.set_title("Decision Tree")
      #写在右下角的数字
      ax.text(array1.max() - .3, array2.min() + .3, ('{:.1f}%'.format(score*100)),
              size=15, horizontalalignment='right') 
      
      #让i继续加一
      i += 1
      
  plt.tight_layout()
  plt.show()
  ```

  

---

### 分类树实例：泰坦尼克号生存预测

#### 代码分解

* 需要导入的库

  ```python
  """导入所需要的库"""
  import pandas as pd
  import numpy as np
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import cross_val_score
  import matplotlib.pyplot as plt
  ```

  

* 导入数据集

  ```python
  """导入数据集，探索数据"""
  data_train = pd.read_csv('./need/Taitanic_data/data.csv',index_col=0)
  data_test = pd.read_csv('./need/Taitanic_data/test.csv',index_col=0)
  
  data = pd.concat([data_train,data_test],axis=0)
  data.head()
  data.info()
  ```

  ​	![20211207VOfRuP](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 VOfRuP.png)

  ![20211207faTpKR](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 faTpKR.png)

* 对数据集进行预处理

  ```python
  """对数据集进行预处理"""
  #删除缺失值过多的列，和观察判断来说和预测的y没有关系的列 
  data.drop(["Cabin","Name","Ticket"],inplace=True,axis=1)
  #处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法 
  data["Age"] = data["Age"].fillna(data["Age"].mean())
  data = data.dropna()
  #将分类变量转换为数值型变量
  #将二分类变量转换为数值型变量 #astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这 个方式可以很便捷地将二分类特征转换为0~1
  data["Sex"] = (data["Sex"]== "male").astype("int")
  #将三分类变量转换为数值型变量
  labels = data["Embarked"].unique().tolist()
  data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
  #查看处理后的数据集 
  data.head()
  ```

  ![20211207cmRwvy](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 cmRwvy.png)

* 提取标签和特征矩阵，分测试集和训练集

  ```python
  """提取标签和特征矩阵，分测试集和训练集"""
  X = data.iloc[:,data.columns != "Survived"]
  y = data.iloc[:,data.columns == "Survived"]
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3)
  #修正测试集和训练集的索引（或者直接reset_index(drop=True,inplace=True)）
  for i in [Xtrain, Xtest, Ytrain, Ytest]:
      i.index = range(i.shape[0])
  #查看分好的训练集和测试集 
  Xtrain.head()
  ```

  ![20211207ylMKCB](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 ylMKCB.png)

* 导入模型，粗略跑一下查看结果

  ```python
  """导入模型，粗略跑一下查看结果"""
  clf = DecisionTreeClassifier(random_state=25)
  clf = clf.fit(Xtrain, Ytrain)
  score_ = clf.score(Xtest, Ytest)
  print('单颗决策树精度',score_)
  score = cross_val_score(clf,X,y,cv=10).mean()
  print('10次交叉验证平均精度',score)
  
  """输出"""
  单颗决策树精度 0.8164794007490637
  10次交叉验证平均精度 0.7739274770173645
  ```

* 在不同max_depth下观察模型的拟合状况

  ```python
  """在不同max_depth下观察模型的拟合状况"""
  tr = [] # 训练集精度
  te = [] # 测试集交叉验证精度
  for i in range(10):
      clf = DecisionTreeClassifier(random_state=25
                                   ,max_depth=i+1
                                   ,criterion="entropy"
                                  )
      clf = clf.fit(Xtrain, Ytrain)
      score_tr = clf.score(Xtrain,Ytrain)
      score_te = cross_val_score(clf,X,y,cv=10).mean()
      tr.append(score_tr)
      te.append(score_te)
  print("测试集交叉验证均值最大值（精度）",max(te))
  plt.figure(figsize=(12,8))
  plt.plot(range(1,11),tr,color="red",label="train")
  plt.plot(range(1,11),te,color="blue",label="test")
  plt.xticks(range(1,11))
  plt.legend()
  plt.show()
  #这里为什么使用“entropy”?因为我们注意到，在最大深度=3的时候，模型拟合不足，在训练集和测试集上的表现接 近，但却都不是非常理想，只能够达到83%左右，所以我们要使用entropy。
  
  
  """输出"""
  测试集交叉验证均值最大值（精度） 0.8177860061287026
  ```

  ​	![202112070tP4hM](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 0tP4hM.jpg)

* 用`网格搜索`调整参数

  ```python
  """用网格搜索调整参数"""
  gini_thresholds = np.linspace(0,0.5,20)
  parameters = {'splitter':('best','random')
                ,'criterion':("gini","entropy")
                ,"max_depth":[*range(1,10)]
                ,'min_samples_leaf':[*range(1,50,5)]
                ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
  }
  clf = DecisionTreeClassifier(random_state=25)
  GS = GridSearchCV(clf, parameters, cv=10)
  GS.fit(Xtrain,Ytrain)
  print('最佳参数',GS.best_params_)
  print('最佳精度',GS.best_score_)
  
  
  """输出"""
  最佳参数 {'criterion': 'entropy'
            , 'max_depth': 9
            , 'min_impurity_decrease': 0.0
            , 'min_samples_leaf': 6
            , 'splitter': 'best'
           }
  最佳精度 0.815284178187404
  ```

  由此可见，网格搜索并非一定比自己调参好，因为网格搜索无法舍弃无用的参数，默认传入的所有参数必须得都选上。

#### 所有代码

* 所有代码

  ```python
  """导入所需要的库"""
  import pandas as pd
  import numpy as np
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import cross_val_score
  import matplotlib.pyplot as plt
  
  """导入数据集，探索数据"""
  data_train = pd.read_csv('./need/Taitanic_data/data.csv',index_col=0)
  data_test = pd.read_csv('./need/Taitanic_data/test.csv',index_col=0)
  
  data = pd.concat([data_train,data_test],axis=0)
  data.head()
  data.info()
  
  
  """对数据集进行预处理"""
  #删除缺失值过多的列，和观察判断来说和预测的y没有关系的列 
  data.drop(["Cabin","Name","Ticket"],inplace=True,axis=1)
  #处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法 
  data["Age"] = data["Age"].fillna(data["Age"].mean())
  data = data.dropna()
  #将分类变量转换为数值型变量
  #将二分类变量转换为数值型变量 #astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这 个方式可以很便捷地将二分类特征转换为0~1
  data["Sex"] = (data["Sex"]== "male").astype("int")
  #将三分类变量转换为数值型变量
  labels = data["Embarked"].unique().tolist()
  data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
  #查看处理后的数据集 
  data.head()
  
  
  """提取标签和特征矩阵，分测试集和训练集"""
  X = data.iloc[:,data.columns != "Survived"]
  y = data.iloc[:,data.columns == "Survived"]
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3)
  #修正测试集和训练集的索引（或者直接reset_index(drop=True,inplace=True)）
  for i in [Xtrain, Xtest, Ytrain, Ytest]:
      i.index = range(i.shape[0])
  #查看分好的训练集和测试集 
  Xtrain.head()
  
  
  """导入模型，粗略跑一下查看结果"""
  clf = DecisionTreeClassifier(random_state=25)
  clf = clf.fit(Xtrain, Ytrain)
  score_ = clf.score(Xtest, Ytest)
  print('单颗决策树精度',score_)
  score = cross_val_score(clf,X,y,cv=10).mean()
  print('10次交叉验证平均精度',score)
  
  
  """在不同max_depth下观察模型的拟合状况"""
  tr = [] # 训练集精度
  te = [] # 测试集交叉验证精度
  for i in range(10):
      clf = DecisionTreeClassifier(random_state=25
                                   ,max_depth=i+1
                                   ,criterion="entropy"
                                  )
      clf = clf.fit(Xtrain, Ytrain)
      score_tr = clf.score(Xtrain,Ytrain)
      score_te = cross_val_score(clf,X,y,cv=10).mean()
      tr.append(score_tr)
      te.append(score_te)
  print("测试集交叉验证均值最大值（精度）",max(te))
  plt.figure(figsize=(12,8))
  plt.plot(range(1,11),tr,color="red",label="train")
  plt.plot(range(1,11),te,color="blue",label="test")
  plt.xticks(range(1,11))
  plt.legend()
  plt.show()
  #这里为什么使用“entropy”?因为我们注意到，在最大深度=3的时候，模型拟合不足，在训练集和测试集上的表现接 近，但却都不是非常理想，只能够达到83%左右，所以我们要使用entropy。
  """用网格搜索调整参数"""
  # gini系数最大为0.5最小为0、信息增益最大为1，最小为0
  gini_thresholds = np.linspace(0,0.5,20)
  parameters = {'splitter':('best','random')
                ,'criterion':("gini","entropy")
                ,"max_depth":[*range(1,10)]
                ,'min_samples_leaf':[*range(1,50,5)]
                ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
  }
  clf = DecisionTreeClassifier(random_state=25)
  GS = GridSearchCV(clf, parameters, cv=10)
  GS.fit(Xtrain,Ytrain)
  print('最佳参数',GS.best_params_)
  print('最佳精度',GS.best_score_)
  ```

  

---

## 决策树优缺点

* 决策树优点
  1. 易于理解和解释，因为树木可以画出来被看见
  2. 需要很少的数据准备。其他很多算法通常都需要数据规范化，需要创建虚拟变量并删除空值等。但请注意，
  sklearn中的决策树模块不支持对缺失值的处理。
  3. 使用树的成本(比如说，在预测数据的时候)是用于训练树的数据点的数量的对数，相比于其他算法，这是
  一个很低的成本。
  4. 能够同时处理数字和分类数据，既可以做回归又可以做分类。其他技术通常专门用于分析仅具有一种变量类
  型的数据集。
  5. 能够处理多输出问题，即含有多个标签的问题，注意与一个标签中含有多种标签分类的问题区别开
  6. 是一个白盒模型，结果很容易能够被解释。如果在模型中可以观察到给定的情况，则可以通过布尔逻辑轻松
      解释条件。相反，在黑盒模型中(例如，在人工神经网络中)，结果可能更难以解释。
  7. 可以使用统计测试验证模型，这让我们可以考虑模型的可靠性。
  8. 即使其假设在某种程度上违反了生成数据的真实模型，也能够表现良好

* 决策树的缺点
  1. 决策树学习者可能创建过于复杂的树，这些树不能很好地推广数据。这称为过度拟合。修剪，设置叶节点所 需的最小样本数或设置树的最大深度等机制是避免此问题所必需的，而这些参数的整合和调整对初学者来说 会比较晦涩
  2. 决策树可能不稳定，数据中微小的变化可能导致生成完全不同的树，这个问题需要通过集成算法来解决。
  3. 决策树的学习是基于贪婪算法，它靠优化局部最优(每个节点的最优)来试图达到整体的最优，但这种做法 不能保证返回全局最优决策树。这个问题也可以由集成算法来解决，在随机森林中，特征和样本会在分枝过 程中被随机采样。
  4. 有些概念很难学习，因为决策树不容易表达它们，例如XOR，奇偶校验或多路复用器问题。
  5. 如果标签中的某些类占主导地位，决策树学习者会创建偏向主导类的树。因此，建议在拟合决策树之前平衡 数据集。

---

## 决策树参数、属性、接口

### 分类树参数列表

* 分类树种的参数列表如下：

  | 参数                       | 参数解释                                                     |
  | :------------------------- | :----------------------------------------------------------- |
  | `criterion`                | 1. 类型：字符型（可不填）<br/>2. 默认：`gini`<br/>3. 作用：衡量分支质量（不纯度）的指标<br/>4. 可选项：<br/>	`gini`：基尼系数<br/>	`entropy`：使用的是信息增益（`information gain`） |
  | `splitter`                 | 1. 类型：字符型（可不填）<br/>2. 默认：`best`<br/>3. 作用：确定每个节点的分支策略<br/>4. 可选项：<br/>	`best`：选择最佳分支<br/>	`random`：使用最佳随机分支 |
  | `max_depth`                | 1. 类型：整数或`None`（可不填）<br/>2. 默认：`None`（树会持续声张直到所有叶子节点不纯度为0,或者直到每个叶子节点所含的样本量都小于参数`min_samples_split`）<br/>3. 作用：树的最大深度 |
  | `min_samples_split`        | 1. 类型：整数或浮点数（可不填）<br/>2. 默认：2<br/>3. 作用：一个中间节点要分支所需要的的最小样本量，若一个节点包含的样本量小于`min_samples_split`,这个节点的分支就不会发生，即，这个节点一定会成为一个叶子节点<br/>4. 可选项：<br/>	输入整数：输入的数字是分支所需的最小样本量<br/>	输入浮点数：比例，对于样本量乘以该浮点数，则是分支所需的最小样本量 |
  | `min_samples_leaf`         | 1. 类型：整数或浮点数（可不填）<br/>2. 默认：1<br/>3. 作用：一个叶子节点要分支所需要的的最小样本量，若一个节点在分支后的每个子节点中，必须要包含至少`min_samples_split`个训练样本，否则分支就不会发生，这个参数可能会使得模型更平滑的效果，尤其是在回归中<br/>4. 可选项：<br/>	输入整数：输入的数字是叶节点存在所需的最小样本量<br/>	输入浮点数：比例，对于样本量乘以该浮点数，叶节点存在所需的最小样本量 |
  | `min_weight_fraction_leaf` | 1. 类型：整数或浮点数（可不填）<br/>2. 默认：0<br/>3. 作用：一个叶节点要存在所需要的权重占输入模型的数据集的总权重的比例。总权重由`fit`接口中的`sample_wieght`参数确定，当`sample_weight`是`None`时，默认所有样本的权重相同 |
  | `max_features`             | 1. 类型：整数、浮点数、字符型、`None`（可不填）<br/>2. 默认：`None`<br/>3. 作用：在做最佳分支时候，考虑的特征个数<br/>4. 可选项：<br/>	输入整数：每个分支都考虑该整数个特征<br/>	输入浮点数：比列，每次分支考虑的特征数目是n_features乘以该比例<br/>	输入’`auto`‘：`n_features`平方根<br/>	输入’`sqrt`‘：`n_features`平方根<br/>	输入’`log2`’：`log2(n_features)`<br/>	输入’`None`‘：`n_features`<br/>注意：如果在限制的`max_features`中，决策树无法找到节点样本上至少一个有效的分支，那对分支的搜索不会停止，决策树将会检查比限制的`max_feautures`数目更多的特征 |
  | `random_state`             | 1. 类型：整数、`None`（可不填）<br/>2. 默认：`None`<br/>3. 作用：随机数种子<br/>4. 可选项：<br/>	输入整数：`random_state`是由随机数生成器生成的随机数种子<br/>	输入`RandomState：random_state`是一个随机数生成器<br/>	输入’`None`‘：随机数生成器回事np。`random`模块中的一个`RandomState`实例 |
  | `max_leaf_nodes`           | 1. 类型：整数、`None`（可不填）<br/>2. 默认：`None`<br/>3. 作用：最大叶节点数量<br/>4. 可选项：<br/>	输入整数：在最佳分枝方式下，以`max_leaf_nodes`为限制来生长树<br/>	输入`None`：没有叶节点数量的限制 |
  | `min_impurity_decrease`    | 1. 类型：浮点数（可不填）<br/>2. 默认：0<br/>3. 作用：当一个节点的分枝后引起的不纯度的降低大于或等于`min_impurity_decrease`中输入的数值，则这个分支会被保留，不会被剪枝<br/>4. 可选项：<br/>	输入整数：在最佳分枝方式下，以`max_leaf_nodes`为限制来生长树<br/>	输入`None`：没有叶节点数量的限制 |
  | `min_impurity_split`       | 1. 类型：浮点数（可不填）<br/>2. 默认：0<br/>3. 作用：放置树生长的阈值之一。若一个节点的不纯度高于`min_impurity_split`,这个节点就会被分枝，否则的话这个节点就只能是叶子节点<br/>备注：0.21版本以上被删除，请使用`min_impurity_decrease` |
  | `class_weight`             | 1. 类型：字典、字典的列表、`balanced`、`None`（可不填）<br/>2. 默认：`None`<br/>3. 作用：与标签相关联的权重，表现方式是（标签的值：权重）<br/>注意：如果指定了`sample_weight`,这些权重将通过fit接口与`sample_weight`相乘 |
  | `presort`                  | 1. 类型：布尔值（可不填）<br/>2. 默认：`False`<br/>3. 作用：是否预先分配数据以价款拟合中最佳分支的发现<br/><br/>备注：大型数据集上使用默认设置决策树时，将这个参数设为True可能会延长训练过程，降低训练速度，当使用较小的数据集或限制树的深度时，<br/>设置这个参数为True可能会加快训练速度 |

---

### 分类树属性列表

* 分类树属性列表如下

  | 属性                   | 属性解释                                                     |
  | ---------------------- | ------------------------------------------------------------ |
  | `classes`              | 1. 输出：数组、列表<br/>2. 结构：标签的数目<br/>3. 解释：所有标签 |
  | `feature_importances_` | 1. 输出：数组<br/>2. 结构：特征的数目（`n_features`）<br/>3. 解释：返回特征重要性，一般是这个特征在多次分枝中产生的信息增益的综合，也被成为“基尼重要性”（`gini importance`） |
  | `max_features_`        | 1. 输出：整数<br/>2. 解释：参数`max_features`的推断值        |
  | `n_classes_`           | 1. 输出：整数或列表<br/>2. 解释：标签类别的数据              |
  | `n_features_`          | 在训练模型（`fit`）时使用的特征个数                          |
  | `n_outputs_`           | 在训练模型（`fit`）时输出的结果的个数                        |
  | `tree_`                | 输出一个可以导出建好的树的结构的端口，通过这个端口，可以访问树的结构和低级属性，包括但是不限于查看：<br/>	1. 二叉树的结构<br/>	2. 每个节点的深度以及它是否是叶子<br/>	3. 使用`decision_path`方法的示例到达的节点<br/>	4. 用`apply`这个接口取样出的叶子<br/>	5. 用于预测样本的规则<br/>	6. 一组样本共享的决策路径 |

  tree_的更多内容可以参考:

  > http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tr ee-plot-unveil-tree-structure-py
  > 

---

### 分类树接口列表

* 分类树接口如下

  ​	![image-20211224002312807](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20211224002312807.png)

---

# 二、集成算法—随机森林

## 机器学习中调参的基本思想

* 调参的方式总是根据数据的状况而定，所 以没有办法一概而论
* 通过画学习曲线，或者网格搜索，我们能够探索到调参边缘（代价可能是训练一次模型要跑三天三夜），但是在现 实中，高手调参恐怕还是多依赖于经验，而这些经验，来源于：
  1. 非常正确的调参思路和方法
  2. 对模型评估指 标的理解
  3. 对数据的感觉和经验
  4. 用洪荒之力去不断地尝试

* 首先来讲讲正确的调参思路。模型调参，第一步是要找准目标：我们要做什么？一般来说，这个目标是提升 某个模型评估指标，比如对于随机森林来说，我们想要提升的是模型在未知数据上的准确率（由score或 oob_score_来衡量）。找准了这个目标，我们就需要思考：模型在未知数据上的准确率受什么因素影响？在机器学 习中，我们用来衡量模型在未知数据上的准确率的指标，叫做泛化误差（Genelization error）

#### 泛化误差

* 当模型在未知数据（测试集或者袋外数据）上表现糟糕时，我们说模型的泛化程度不够，泛化误差大，模型的效果 不好。泛化误差受到模型的结构（复杂度）影响。看下面这张图，它准确地描绘了泛化误差与模型复杂度的关系， 当模型太复杂，模型就会过拟合，泛化能力就不够，所以泛化误差大。当模型太简单，模型就会欠拟合，拟合能力 就不够，所以误差也会大。只有当模型的复杂度刚刚好的才能够达到泛化误差最小的目标。

  ![1](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211209000631.png)

  那模型的复杂度与我们的参数有什么关系呢？对树模型来说，树越茂盛，深度越深，枝叶越多，模型就越复杂。所 以树模型是天生位于图的右上角的模型，随机森林是以树模型为基础，所以随机森林也是天生复杂度高的模型。随 机森林的参数，都是向着一个目标去：减少模型的复杂度，把模型往图像的左边移动，防止过拟合。当然了，调参 没有绝对，也有天生处于图像左边的随机森林，所以调参之前，我们要先判断，模型现在究竟处于图像的哪一边。 泛化误差的背后其实是“偏差-方差困境”，原理十分复杂，无论你翻开哪一本书，你都会看见长篇的数学论证和每个 字都能看懂但是连在一起就看不懂的文字解释。在下一节偏差vs方差中，我用最简单易懂的语言为大家解释了泛化 误差背后的原理，大家选读。那我们只需要记住这四点：

  1. 模型太复杂或者太简单，都会让泛化误差高，我们追求的是位于中间的平衡点 
  2. 模型太复杂就会过拟合，模型太简单就会欠拟合 
  3. 对树模型和树的集成模型来说，树的深度越深，枝叶越多，模型越复杂 
  4. 树模型和树的集成模型的目标，都是减少模型复杂度，把模型往图像的左边移动

  那具体每个参数，都如何影响我们的复杂度和模型呢？我们一直以来调参，都是在学习曲线上轮流找最优值，盼望 能够将准确率修正到一个比较高的水平。然而我们现在了解了随机森林的调参方向：降低复杂度，我们就可以将那 些对复杂度影响巨大的参数挑选出来，研究他们的单调性，然后专注调整那些能最大限度让复杂度降低的参数。对 于那些不单调的参数，或者反而会让复杂度升高的参数，我们就视情况使用，大多时候甚至可以退避。基于经验， 我对各个参数对模型的影响程度做了一个排序。在我们调参的时候，大家可以参考这个顺序。

  | 参数                 | 对模型在未知数据上的评估性能的影响                           | 影响程度   |
  | -------------------- | ------------------------------------------------------------ | ---------- |
  | `n_estimators`       | 提升至平稳，n_estimators↑，不影响单个模型的复杂度            | ⭐⭐⭐⭐       |
  | `max_depth`          | 有增有减，默认最大深度，即最高复杂度，向复杂度降低的方向调参 max_depth↓，模型更简单，且向图像的左边移动 | ⭐⭐⭐        |
  | `min_samples _leaf`  | 有增有减，默认最小限制1，即最高复杂度，向复杂度降低的方向调参 min_samples_leaf↑，模型更简单，且向图像的左边移动 | ⭐⭐         |
  | `min_samples _split` | 有增有减，默认最小限制2，即最高复杂度，向复杂度降低的方向调参 min_samples_split↑，模型更简单，且向图像的左边移动 | ⭐⭐         |
  | `max_features`       | 有增有减，默认auto，是特征总数的开平方，位于中间复杂度，既可以 向复杂度升高的方向，也可以向复杂度降低的方向调参 max_features↓，模型更简单，图像左移 max_features↑，模型更复杂，图像右移 max_features是唯一的，既能够让模型更简单，也能够让模型更复杂的参 数，所以在调整这个参数的时候，需要考虑我们调参的方向 | ⭐          |
  | `criterion`          | 有增有减，一般使用gini                                       | 看具体情况 |

  有了以上的知识储备，我们现在也能够通过参数的变化来了解，模型什么时候到达了极限，当复杂度已经不能再降 低的时候，我们就不必再调整了，因为调整大型数据的参数是一件非常费时费力的事。除了学习曲线和网格搜索， 我们现在有了基于对模型和正确的调参思路的“推测”能力，这能够让我们的调参能力更上一层楼。

#### 偏差vs方差

* 一个集成模型( f )在未知数据( D )上的 泛化误差E( f ; D )，由方差(var) 偏差( bias )和噪声($\varepsilon$)共同决定
  $$
  E(f;D) = bias^2(x)+var(x)+\varepsilon ^2
  $$
  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211209001856.png)

* 上面的图像中，每个点就是集成算法中的一个基评估器产生的预测值。红色虚线代表着这些预测值的均值， 而蓝色的线代表着数据本来的面貌。

  	1. 偏差：偏差：模型的预测值与真实值之间的差异，即每一个红点到蓝线的距离。在集成算法中，每个基评估器都会有 自己的偏差，集成评估器的偏差是所有基评估器偏差的均值。模型越精确，偏差越低。
  	2. 方差：反映的是模型每一次输出结果与模型预测值的平均水平之间的误差，即每一个红点到红色虚线的距离， 衡量模型的稳定性。模型越稳定，方差越低。

* 其中偏差衡量模型是否预测得准确，`偏差越小，模型越“准”`；

  而方差衡量模型每次预测的结果是否接近，即是说`方 差越小，模型越“稳”`；

  噪声是机器学习无法干涉的部分。一个好的模 型，要对大多数未知数据都预测得”准“又”稳“。即是说，当偏差和方差都很低的时候，模型的泛化误差就小，在未 知数据上的准确率就高。

* 通常来说，方差和偏差有一个很大，泛化误差都会很大。然而，方差和偏差是此消彼长的，不可能同时达到最小 值。这个要怎么理解呢？来看看下面这张图

  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211209002512.png)

  从图上可以看出，`模型复杂度大的时候，方差高，偏差低`。偏差低，就是要求模型要预测得“准”。模型就会更努力 去学习更多信息，会具体于训练数据，这会导致，模型在一部分数据上表现很好，在另一部分数据上表现却很糟 糕。模型泛化性差，在不同数据上表现不稳定，所以方差就大。而要尽量学习训练集，模型的建立必然更多细节， 复杂程度必然上升。所以，复杂度高，方差高，总泛化误差高。

  相对的，`复杂度低的时候，方差低，偏差高`。方差低，要求模型预测得“稳”，泛化性更强，那对于模型来说，它就 不需要对数据进行一个太深的学习，只需要建立一个比较简单，判定比较宽泛的模型就可以了。结果就是，模型无 法在某一类或者某一组数据上达成很高的准确度，所以偏差就会大。所以，复杂度低，偏差高，总泛化误差高。

* 我们调参的目标是，达到方差和偏差的完美平衡！虽然方差和偏差不能同时达到最小值，但他们组成的泛化误差却 可以有一个最低点，而我们就是要寻找这个最低点。对复杂度大的模型，要降低方差，对相对简单的模型，要降低 偏差。随机森林的基评估器都拥有较低的偏差和较高的方差，因为决策树本身是预测比较”准“，比较容易过拟合的 模型，装袋法本身也要求基分类器的准确率必须要有50%以上。所以以随机森林为代表的装袋法的训练过程旨在降 低方差，即降低模型复杂度，所以随机森林参数的默认设定都是假设模型本身在泛化误差最低点的右边。

* 所以，我们`在降低复杂度的时候，本质其实是在降低随机森林的方差`，随机森林所有的参数，也都是朝着降低方差 的目标去。有了这一层理解，我们对复杂度和泛化误差的理解就更上一层楼了，对于我们调参，也有了更大的帮助。

---

## 集成算法概述

* `集成学习(ensemble learning)`是时下非常流行的机器学习算法，它本身不是一个单独的机器学习算法，而是通 过在数据上构建多个模型，集成所有模型的建模结果。基本上所有的机器学习领域都可以看到集成学习的身影，在 现实中集成学习也有相当大的作用，它可以用来做市场营销模拟的建模，统计客户来源，保留和流失，也可用来预 测疾病的风险和病患者的易感性。在现在的各种算法竞赛中，随机森林，梯度提升树(GBDT)，Xgboost等集成 算法的身影也随处可见，可见其效果之好，应用之广。

* 集成算法的`目标`：

  考虑多个评估器的建模结果，汇总之后得到一个综合的结果，以此来获得比单个模型更好的回归或分类表现。

* 多个模型集成成为的模型叫做集成评估器(ensemble estimator)，组成集成评估器的每个模型都叫做基评估器 (base estimator)。通常来说，有三类集成算法:

  1. 装袋法(Bagging)：构建多个互相独立的评估器，对其预测进行平均或多数表决。代表模型随机森林

  2. 提升法(Boosting)：基评估器是相关的，按顺序一一构建。其核心思想是结合弱评估器的力量一次次对难以评估的样本 进行预测，从而构成一个强评估器。提升法的代表模型有Adaboost和梯度提升树。

  3. stacking。

     ![20211207oDkIJx](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 oDkIJx.png)

---

## sklearn中的集成算法

* `sklearn`中的集成算法模块`ensemble`

  | 类                                  | 类的功能                                |
  | ----------------------------------- | --------------------------------------- |
  | ensemble.AdaBoostClassifier         | AdaBoost分类                            |
  | ensemble.AdaBoostRegressor          | AdaBoost回归                            |
  | ensemble.BaggingClassifier          | 袋装分类器                              |
  | ensemble.BaggingRegressor           | 袋装回归器                              |
  | ensemble.ExtraTreesClassifier       | Extra-trees分类(超树，极端随机树)       |
  | ensemble.ExtraTreesRegressor        | Extra-trees回归                         |
  | ensemble.GradientBoostingClassifier | 梯度提升分类                            |
  | ensemble.GradientBoostingRegressor  | 梯度提升回归                            |
  | ensemble.IsolationForest            | 孤立森林                                |
  | ensemble.RandomForestClassifier     | 随机森林分类                            |
  | ensemble.RandomForestRegressor      | 随机森林回归                            |
  | ensemble.RandomTreesEmbedding       | 完全随机树的集成                        |
  | ensemble.VotingClassifier           | 用于不合适估算器的软投票/多数规则分类器 |

  集成算法中，有一半以上都是树的集成模型，可以想见决策树在集成中必定是有很好的效果。

## RandomForestClassifier

* `sklearn.ensemble.RandomForestClassifier`

  ```python
  sklearn.ensemble.RandomForestClassifier (n_estimators=’10’
                                           , criterion=’gini’
                                           , max_depth=None
                                           , min_samples_split=2
                                           , min_samples_leaf=1
                                           , min_weight_fraction_leaf=0.0
                                           , max_features=’auto’
                                           , max_leaf_nodes=None
                                           , min_impurity_decrease=0.0
                                           , min_impurity_split=None
                                           , bootstrap=True
                                           , oob_score=False
                                           , n_jobs=None
                                           , random_state=None
                                           , verbose=0
                                           , warm_start=False
                                           , class_weight=None
                                          )
  ```

### 控制基评估器的参数

* 参数

  | 参数                  | 含义                                                         |
  | --------------------- | ------------------------------------------------------------ |
  | criterion             | 不纯度的衡量指标，有gini系数和信息熵entropy（实际用的是信息增益）两种选择 |
  | max_depth             | 树的最大深度，超过最大深度的树会被剪掉                       |
  | min_samples_leaf      | 一个节点在分支后的每个子节点都必须包含至少min_samples_leaf个训练样本，否则分支就不会发生 |
  | min_samples_split     | 一个节点必须哟啊包含至少min_samples_split个训练样本，这个节点才允许被分支，否则分支就不会发生 |
  | max_features          | 限制分支时考虑的特征个数，超过限制个数的特征会被舍弃，默认值为总特征个数开平方取整。 |
  | min_impurity_decrease | 限制信息增益的大小，信息增益小于设定数值的分支不会发生       |

  单个决策树的准确率越高，随机森林的准确率也会越高，因为装袋法是依赖于平均值或少数服从多数的原则来决定集成结果的。

### n_estimators(随机森林与决策树交叉验证对比、n_estimators学习曲线)

* 这是森林中树木的数量，即基评估器的数量。`这个参数对随机森林模型的精确性影响是单调的，n_estimators越 大，模型的效果往往越好。但是相应的，任何模型都有决策边界，n_estimators达到一定的程度之后，随机森林的 精确性往往不在上升或开始波动`，并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越 长。对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。
  n_estimators的默认值在现有版本的sklearn中是10，但是在0.22版本中，这个默认值被修正为 100。这个修正显示出了使用者的调参倾向:要更大的n_estimators。

#### 代码分解

* 导入需要的包、数据集

  ```python
  """导入需要的包、数据集"""
  import pandas as pd
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import cross_val_score
  from matplotlib import pyplot as plt
  from tqdm import notebook
  
  plt.rcParams['font.sans-serif'] = ['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.rcParams["font.family"] = 'Arial Unicode MS'
  %matplotlib inline
  
  
  wine = load_wine()
  data = pd.DataFrame(wine.data,columns=wine.feature_names)
  data['y'] = wine.target
  data.head()
  ```

  ​	![20211207RMvqKP](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 RMvqKP.png)

* 决策树、随机森林建模

  ```python
  """决策树、随机森林建模"""
  X_train,X_test,y_train,y_test = train_test_split(wine.data
                                                   ,wine.target
                                                   ,test_size=.3
                                                  )
  # 单颗决策树
  clf = DecisionTreeClassifier(random_state=0)
  #随机森林
  rfc = RandomForestClassifier(random_state=0)
  clf = clf.fit(X_train,y_train)
  rfc = rfc.fit(X_train,y_train)
  score_clf = clf.score(X_test,y_test)
  score_rfc = rfc.score(X_test,y_test)
  
  print(f'Single Tree:   {score_clf}')
  print(f'Random Forest:   {score_rfc}')
  
  """输出"""
  Single Tree:   0.9074074074074074
  Random Forest:   0.9814814814814815
  ```



* 画出随机森林和决策树在一组十折交叉验证下的效果对比

  ```python
  """画出随机森林和决策树在一组十折交叉验证下的效果对比"""
  # 单颗决策树、交叉验证
  clf = DecisionTreeClassifier()
  clf_s = cross_val_score(clf,wine.data,wine.target,cv=10)
  # 随机森林、交叉验证
  rfc = RandomForestClassifier(n_estimators=25)
  rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10)
  
  plt.figure(figsize=(12,8))
  plt.plot(range(1,11),rfc_s,label = "RandomForest")
  plt.plot(range(1,11),clf_s,label = "Decision Tree")
  plt.xticks(rotation='-45')
  plt.ylabel('精度(整体准确率)')
  plt.title('随机森林和决策树在一组十折交叉验证下的效果对比')
  plt.legend()
  plt.show()
  #====================一种更加有趣也更简单的写法===================#
  """
  label = "RandomForest"
  for model in [RandomForestClassifier(n_estimators=25),DecisionTreeClassifier()]:
      score = cross_val_score(model,wine.data,wine.target,cv=10)
      print("{}:".format(label)),print(score.mean())
      plt.plot(range(1,11),score,label = label)
      plt.legend()
      label = "DecisionTree"
  """
  ```

  ​	![20211207wl7uIq](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 wl7uIq.jpg)

  可以看出，单棵决策树的效果是不如（小于等于）随机森林的



* 画出随机森林和决策树在十组十折交叉验证下的效果对比

  ```python
  """画出随机森林和决策树在十组十折交叉验证下的效果对比"""
  rcf_1 = [] # 随机森林十组十折交叉验证均值记录
  clf_1 = [] # 单颗决策树十组十折交叉验证均值记录
  
  for i in range(10):
      rcf = RandomForestClassifier(n_estimators=25)
      rcf_s = cross_val_score(rcf,wine.data,wine.target,cv=10).mean()
      rcf_1.append(rcf_s)
      
      clf = DecisionTreeClassifier()
      clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
      clf_1.append(clf_s)
  
  plt.figure(figsize=(12,8))
  plt.plot([f"第{i}组" for i in range(1,11)],rcf_1,label='Random Forest')
  plt.plot([f"第{i}组" for i in range(1,11)],clf_1,label='Decision Tree')
  plt.ylabel('10折交叉验证精度平均值')
  plt.title('随机森林和决策树在十组十折交叉验证下的效果对比')
  plt.legend()
  plt.show()
  ```

  ​	![202112075E3MlK](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 5E3MlK.jpg)



* `n_estimators`学习曲线

  ```python
  """n_estimators学习曲线"""
  superpa = [] # 记录不同基分类器数量下，随机森林交叉验证平均值
  for i in notebook.tqdm(range(1,201)):
      rfc = RandomForestClassifier(n_estimators=i,n_jobs=-1)
      rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
      superpa.append(rfc_s)
  print(f"随机森林交叉验证精度最大值：{max(superpa)}")
  print(f'随机森林精度对大值对应的基分类器数量：',superpa.index(max(superpa)))
  plt.figure(figsize=(12,8)) 
  plt.plot(range(1,201),superpa)
  plt.xlabel('基分类器数量')
  plt.ylabel('10折交叉验证精度平均值')
  plt.title('n_estimators学习曲线')
  plt.show()
  ```

  ​	![202112070fITc0](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 07 0fITc0.jpg)

  可见随着基分类器数量增加，随机森林准确度上升，当到达一定数量后趋于平缓（例子中是约15左右）。

#### 所有代码

* 所有代码

  ```python
  """导入需要的包、数据集"""
  import pandas as pd
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import cross_val_score
  from matplotlib import pyplot as plt
  from tqdm import notebook
  
  plt.rcParams['font.sans-serif'] = ['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.rcParams["font.family"] = 'Arial Unicode MS'
  %matplotlib inline
  
  
  
  wine = load_wine()
  data = pd.DataFrame(wine.data,columns=wine.feature_names)
  data['y'] = wine.target
  data.head()
  
  """决策树、随机森林建模"""
  X_train,X_test,y_train,y_test = train_test_split(wine.data
                                                   ,wine.target
                                                   ,test_size=.3
                                                  )
  # 单颗决策树
  clf = DecisionTreeClassifier(random_state=0)
  #随机森林
  rfc = RandomForestClassifier(random_state=0)
  clf = clf.fit(X_train,y_train)
  rfc = rfc.fit(X_train,y_train)
  score_clf = clf.score(X_test,y_test)
  score_rfc = rfc.score(X_test,y_test)
  
  print(f'Single Tree:   {score_clf}')
  print(f'Random Forest:   {score_rfc}')
  
  """画出随机森林和决策树在一组十折交叉验证下的效果对比"""
  # 单颗决策树、交叉验证
  clf = DecisionTreeClassifier()
  clf_s = cross_val_score(clf,wine.data,wine.target,cv=10)
  # 随机森林、交叉验证
  rfc = RandomForestClassifier(n_estimators=25)
  rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10)
  
  plt.figure(figsize=(12,8))
  plt.plot(range(1,11),rfc_s,label = "RandomForest")
  plt.plot(range(1,11),clf_s,label = "Decision Tree")
  plt.xticks(rotation='-45')
  plt.ylabel('精度(整体准确率)')
  plt.title('随机森林和决策树在一组十折交叉验证下的效果对比')
  plt.legend()
  plt.show()
  #====================一种更加有趣也更简单的写法===================#
  """
  label = "RandomForest"
  for model in [RandomForestClassifier(n_estimators=25),DecisionTreeClassifier()]:
      score = cross_val_score(model,wine.data,wine.target,cv=10)
      print("{}:".format(label)),print(score.mean())
      plt.plot(range(1,11),score,label = label)
      plt.legend()
      label = "DecisionTree"
  """
  
  
  
  """画出随机森林和决策树在十组十折交叉验证下的效果对比"""
  rcf_1 = [] # 随机森林十组十折交叉验证均值记录
  clf_1 = [] # 单颗决策树十组十折交叉验证均值记录
  
  for i in range(10):
      rcf = RandomForestClassifier(n_estimators=25)
      rcf_s = cross_val_score(rcf,wine.data,wine.target,cv=10).mean()
      rcf_1.append(rcf_s)
      
      clf = DecisionTreeClassifier()
      clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
      clf_1.append(clf_s)
  
  plt.figure(figsize=(12,8))
  plt.plot([f"第{i}组" for i in range(1,11)],rcf_1,label='Random Forest')
  plt.plot([f"第{i}组" for i in range(1,11)],clf_1,label='Decision Tree')
  plt.ylabel('10折交叉验证精度平均值')
  plt.title('随机森林和决策树在十组十折交叉验证下的效果对比')
  plt.legend()
  plt.show()
  
  
  """n_estimators学习曲线"""
  superpa = [] # 记录不同基分类器数量下，随机森林交叉验证平均值
  for i in notebook.tqdm(range(1,201)):
      rfc = RandomForestClassifier(n_estimators=i,n_jobs=-1)
      rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
      superpa.append(rfc_s)
  print(f"随机森林交叉验证精度最大值：{max(superpa)}")
  print(f'随机森林精度对大值对应的基分类器数量：',superpa.index(max(superpa)))
  plt.figure(figsize=(12,8)) 
  plt.plot(range(1,201),superpa)
  plt.xlabel('基分类器数量')
  plt.ylabel('10折交叉验证精度平均值')
  plt.title('n_estimators学习曲线')
  plt.show()
  ```

---

### random_state(随机森林中控制一群树)

* 随机森林的本质是一种装袋集成算法(bagging)，装袋集成算法是对基评估器的预测结果进行平均或用多数表决 原则来决定集成评估器的结果。在刚才的红酒例子中，我们建立了25棵树，对任何一个样本而言，平均或多数表决 原则下，当且仅当有13棵以上的树判断错误的时候，随机森林才会判断错误。单独一棵决策树对红酒数据集的分类 准确率在0.85上下浮动，假设一棵树判断错误的可能性为0.2($\varepsilon$)，那13棵树以上都判断错误的可能性是:
  $$
  e_{{random_forest }}=\sum_{i=13}^{25} C_{25}^{i} \varepsilon^{i}(1-\varepsilon)^{25-i}=0.000369
  $$
  其中，i是判断错误的次数，也是判错的树的数量，ε是一棵树判断错误的概率，(1-ε)是判断正确的概率，共判对 25-i次。采用组合，是因为25棵树中，有任意i棵都判断错误。

  ```python
  import numpy as np
  from scipy.special import comb
  
  
  np.array([comb(25,i)*(0.2**i)*((1-0.2)**(25-i)) for i in range(13,26)]).sum()
  ```

  可见，判断错误的几率非常小，这让随机森林在红酒数据集上的表现远远好于单棵决策树。

  那现在就有一个问题了:我们说袋装法服从多数表决原则或对基分类器结果求平均，这即是说，我们默认森林中的 每棵树应该是不同的，并且会返回不同的结果。设想一下，如果随机森林里所有的树的判断结果都一致(全判断对 或全判断错)，那随机森林无论应用何种集成原则来求结果，都应该无法比单棵决策树取得更好的效果才对。但我 们使用了一样的类DecisionTreeClassifier，一样的参数，一样的训练集和测试集，为什么随机森林里的众多树会有 不同的判断结果呢？

  问到这个问题，可能就会想到了:sklearn中的分类树DecisionTreeClassifier自带随机性，所以随机森 林中的树天生就都是不一样的。我们在讲解分类树时曾提到，决策树从最重要的特征中随机选择出一个特征来进行 分枝，因此每次生成的决策树都不一样，这个功能由参数random_state控制。

  `随机森林中其实也有random_state，用法和分类树中相似，只不过在分类树中，一个random_state只控制生成一 棵树，而随机森林中的random_state控制的是生成森林的模式，而非让一个森林中只有一棵树。`

  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_wine
  
  wine = load_wine()
  
  """random_state = 2"""
  rfc = RandomForestClassifier(n_estimators=10,random_state=2)
  rfc = rfc.fit(wine.data, wine.target)
  #随机森林的重要属性之一:estimators，查看森林中树的状况 rfc.estimators_[0].random_state
  for i in range(len(rfc.estimators_)):
      print(rfc.estimators_[i].random_state)
      
  """输出"""
  1872583848
  794921487
  111352301
  1853453896
  213298710
  1922988331
  1869695442
  2081981515
  1805465960
  1376693511
  
  
  """random_state = 3"""
  rfc = RandomForestClassifier(n_estimators=10,random_state=3)
  rfc = rfc.fit(wine.data, wine.target)
  #随机森林的重要属性之一:estimators，查看森林中树的状况 rfc.estimators_[0].random_state
  for i in range(len(rfc.estimators_)):
      print(rfc.estimators_[i].random_state)
      
  """输出"""
  218175338
  303761048
  893988089
  1460070019
  1249426360
  521102280
  46504192
  297689877
  1687694333
  1877166739
  ```

  我们可以观察到，当random_state固定时，随机森林中生成是一组固定的树，但每棵树依然是不一致的，这是 用”随机挑选特征进行分枝“的方法得到的随机性。并且我们可以证明，`当这种随机性越大的时候，袋装法的效果一 般会越来越好`。用袋装法集成时，基分类器应当是相互独立的，是不相同的。
  `但这种做法的局限性是很强的，当我们需要成千上万棵树的时候，数据不一定能够提供成千上万的特征来让我们构 筑尽量多尽量不同的树。因此，除了random_state。我们还需要其他的随机性。`

### bootstrap、oob_score

#### 有放回抽样、bootstrap

* 要让基分类器尽量都不一样，一种很容易理解的方法是使用不同的训练集来进行训练，而袋装法正是通过有放回的 随机抽样技术来形成不同的训练数据，`bootstrap`就是用来控制抽样技术的参数。

* `bootstrap`参数默认True，代表采用这种又放回的随机抽样技术，通常这个参数不会被我们设置为False。

* 然而有放回抽样也会有自己的问题。由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能 被忽略，一般来说，自助集大约平均会包含63%的原始数据。因为每一个样本被抽到某个自助集中的概率为：
  $$
  1-(1-\frac{1}{n })^n \\
  $$

  $$
  \lim_{n\rightarrow \infty}\left[1-(1-\frac{1}{n })^n\right] \quad = \quad 1-\frac{1}{e}
  $$

  当n足够大时，这个概率收敛于1-(1/e)，约等于0.632。

#### 袋外数据、oob、our of bag data

* `bootstrap`会使得约37%的训练集数据被浪费掉，没有参与建模，这些数据被称为袋外 数据（out of bag data,简写为oob）。

* 除了我们最开始就划分好的测试集之外，这些数据也可 以被用来作为集成算法的测试集。也就是说，在使用随机森林时，我们可以不划分测试集和训练集，只需要用袋外 数据来测试我们的模型即可。
* 当然，这也不是绝对的，当n和n_estimators都不够大的时候，很可能就没有数据掉 落在袋外，自然也就无法使用oob数据来测试模型了。
* 如果希望用袋外数据来测试，则需要在实例化时就将oob_score这个参数调整为True，训练完毕之后，我们可以用 随机森林的另一个重要属性：oob_score_来查看我们的在袋外数据上测试的结果。

#### 实例

* 代码

  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_wine
  
  wine = load_wine()
  # 无需划分训练集和测试集
  rfc = RandomForestClassifier(n_estimators=25,oob_score=True)
  rfc = rfc.fit(wine.data,wine.target)
  
  # 重要属性oob_score_
  print('训练集精度：',rfc.score(wine.data,wine.target))
  print('测试集（袋外数据）精度',rfc.oob_score_)
  
  """输出"""
  训练集精度： 1.0
  测试集（袋外数据）精度 0.9662921348314607
  ```

### 其他重要属性和接口

* 属性
  1. `.estimators`
  2. `.oob_score`
  3. `.feature_importances`

* 常用接口
  1. `apply`
  2. `fit`
  3. `predict`
  4. `score`
* 除此之外，还需要注 意随机森林的`predict_proba`接口，这个接口返回每个测试样本对应的被分到每一类标签的概率，标签有几个分类 就返回几个概率。如果是二分类问题，则predict_proba返回的数值大于0.5的，被分为1，小于0.5的，被分为0。 

* `注意`
  1. 传统的随机森林是利用袋装法中的规则，平均或少数服从多数来决定集成的结果
  2. sklearn中的随机森林是平均 每个样本对应的predict_proba返回的概率，得到一个·`平均概率`·，从而决定测试样本的分类

## Bagging的必要条件

* 必要条件
  1. 要求基评估器要尽量独立
  2. 基分类器的判断准确率至少要超过随机分类器。

* 随机森林准确率公式（假设使用25个基分类器）
  $$
  e_{{random_forest }}=\sum_{i=13}^{25} C_{25}^{i} \varepsilon^{i}(1-\varepsilon)^{25-i}
  $$
  基于上面的公式，用下面的代码画出基分类器的误差率 $\varepsilon$ 和随机森林的误差率之间的图像。

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  from scipy.special import comb
  
  %matplotlib inline
  x = np.linspace(0,1,20) # 创建一个基分类器的误差率列表，从0到1之间取等长的20个点
  y = [] # 记录随机森林的误差率
  for epsilon in np.linspace(0,1,20):
      E = np.array([comb(25,i)*(epsilon**i)*((1-epsilon)**(25-i))
                    for i in range(13,26)]).sum()
      y.append(E)
  plt.figure(figsize=(12,8))
  plt.plot(x,y,"o-",label="when estimators are different")
  plt.plot(x,x,"--",color="red",label="if all estimators are same")
  plt.xlabel("individual estimator's error")
  plt.ylabel("RandomForest's error")
  plt.legend()
  plt.show()
  
  ```

  ​	![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/基分类器的误差率ε和随机森林的误差率之间的图像.png)

  可以从图像上看出，当基分类器的误差率小于0.5，即准确率大于0.5时，集成的效果是比基分类器要好的。相反， 当基分类器的误差率大于0.5，袋装的集成算法就失效了。所以在使用随机森林之前，一定要检查，用来组成随机 森林的分类树们是否都有至少50%的预测正确率。

## RandomForestRegressor

* `sklearn.ensemble.RandomForestRegressor`

  ```python
  sklearn.ensemble.RandomForestRegressor (n_estimators=’warn’
                                          , criterion=’mse’
                                          , max_depth=None
                                          , min_samples_split=2
                                          , min_samples_leaf=1
                                          , min_weight_fraction_leaf=0.0
                                          , max_features=’auto’
                                          , max_leaf_nodes=None
                                          , min_impurity_decrease=0.0
                                          , min_impurity_split=None
                                          , bootstrap=True
                                          , oob_score=False
                                          , n_jobs=None
                                          , random_state=None
                                          , verbose=0
                                          , warm_start=False
                                         )
  ```

### criterion

* 回归树衡量分枝质量的指标，支持的标准有三种： 

  1. 输入`mse`使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为 特征选择的标准，这种方法通过使用叶子节点的均值来最小化L2损失 

  2. 输入`friedman_mse`使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差

  3. 输入`mae`使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失
     $$
     M S E=\frac{1}{N} \sum_{i=1}^{N}\left(f_{i}-y_{i}\right)^{2}
     $$

  其中N是样本数量，i是每一个数据样本，fi是模型回归出的数值，yi是样本点i实际的数值标签。所以MSE的本质， 其实是样本真实数据与回归结果的差异。在回归树中，MSE不只是我们的分枝质量衡量指标，也是我们最常用的衡 量回归树回归质量的指标，当我们在使用交叉验证，或者其他方式获取回归树的结果时，我们往往选择均方误差作 为我们的评估（在分类树中这个指标是score代表的预测准确率）。在回归中，我们追求的是，MSE越小越好。 然而，回归树的接口score返回的是R平方，并不是MSE。R平方被定义如下：
  $$
  \begin{array}{c}
  R^{2}=1-\frac{u}{v} \\
  u=\sum_{i=1}^{N}\left(f_{i}-y_{i}\right)^{2} \quad v=\sum_{i=1}^{N}\left(y_{i}-\hat{y}\right)^{2}
  \end{array}
  $$
  其中u是残差平方和（MSE * N），v是总平方和，N是样本数量，i是每一个数据样本，fi是模型回归出的数值，yi 是样本点i实际的数值标签。y帽是真实数值标签的平均数。R平方可以为正为负（如果模型的残差平方和远远大于 模型的总平方和，模型非常糟糕，R平方就会为负），而均方误差永远为正。 值得一提的是，虽然均方误差永远为正，但是sklearn当中使用均方误差作为评判标准时，却是计算”负均方误 差“（neg_mean_squared_error）。这是因为sklearn在计算模型评估指标的时候，会考虑指标本身的性质，均 方误差本身是一种误差，所以被sklearn划分为模型的一种损失(loss)，因此在sklearn当中，都以负数表示。真正的 均方误差MSE的数值，其实就是neg_mean_squared_error去掉负号的数字。

### 其他重要属性和接口

* 最重要的属性和接口，都与随机森林的分类器相一致，还是apply, fit, predict和score最为核心。

* `注意`: 随 机森林回归并**没有predict_proba这个接口**，因为对于回归来说，并不存在一个样本要被分到某个类别的概率问 题，因此没有predict_proba这个接口。

* 随机森林回归用法

  ```python
  from sklearn.datasets import load_boston
  from sklearn.model_selection import cross_val_score
  from sklearn.ensemble import RandomForestRegressor
  import sklearn
  
  
  boston = load_boston()
  regressor = RandomForestRegressor(n_estimators=100,random_state=0)
  cross_val_score(regressor, boston.data, boston.target, cv=10
                 ,scoring = "neg_mean_squared_error")
  """输出"""
  array([-10.60400153,  -5.34859049,  -5.00482902, -21.30948927,
         -12.21354202, -18.50599124,  -6.89427068, -93.92849386,
         -29.91458572, -15.1764633 ])
  ```

  返回十次交叉验证的结果，注意在这里，如果不填写`scoring = "neg_mean_squared_error"`，交叉验证默认的模型 衡量指标是R方，因此交叉验证的结果可能有正也可能有负。而如果写上scoring，则衡量标准是负MSE，交叉验 证的结果只可能为负。

---

## 随机森林实例

### 用随机森林回归填补缺失值

* 我们从现实中收集的数据，几乎不可能是完美无缺的，往往都会有一些缺失值。面对缺失值，很多人选择的方式是 直接将含有缺失值的样本删除，这是一种有效的方法，但是有时候填补缺失值会比直接丢弃样本效果更好，即便我 们其实并不知道缺失值的真实样貌。在	`sklearn`中，我们可以使用`sklearn.impute.SimpleImputer`(0.20版本以上)来轻松地将**均 值**，**中值**，或者其他最常用的数值填补到数据中，在这个案例中，我们将使用均值，0，和随机森林回归来填补缺 失值，并验证四种状况下的拟合状况，找出对使用的数据集来说最佳的缺失值填补方法。

#### 代码分解

* 导入所需要的库

  ```python
  """导入需要的库"""
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.datasets import load_boston
  from sklearn.impute import SimpleImputer
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import cross_val_score
  from tqdm import tqdm
  ```

*  以波士顿房价数据集为例子，导入完整的数据集并探索

  ```python
  datasets = load_boston()
  datasets.data.shape #总共506*13=6578个数据
  
  X_full,y_full = datasets.data,datasets.target # 完整数据集
  n_samples = X_full.shape[0] # 样本量
  n_features = X_full.shape[1] # 特征量
  ```

* 为完整数据集放入缺失值

  ```python
  """为完整数据集放入缺失值"""
  #首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那总共就要有3289个数据缺失
  rng = np.random.RandomState(0)
  missing_rate = 0.5
  n_missing_samples = int(np.floor(n_samples * n_features * missing_rate)) # 3289
  #np.floor向下取整，返回.0格式的浮点数,所以需要将其转换成整数
  
  # 创建0到n_features（这里是13）之间，长度为n_missing_samples（这里是3289）的数组,用来当做列索引
  missing_features = rng.randint(0,n_features,n_missing_samples)
  # 创建0到n_samples（这里是506）之间，长度为n_missing_samples（这里是3289）的数组,用来当做行索引
  missing_samples = rng.randint(0,n_samples,n_missing_samples)
  
  #我们现在采样了3289个数据，远远超过我们的样本量506，所以我们使用随机抽取的函数randint。但如果我们需要的数据量小于我们的样本量506，那我们可以采用np.random.choice来抽样，choice会随机抽取不重复的随机数，因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中
  
  X_missing = X_full.copy()
  y_missing = y_full.copy()
  
  # 将随机抽取的行列，置位Nan
  X_missing[missing_samples,missing_features] = np.nan
  #转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度快到拯救人生，但是在索引等功能上却不如pandas来得好用
  X_missing = pd.DataFrame(X_missing)
  ```

* 使用0和均值填补缺失值

  ```python
  """使用0和均值填补缺失值"""
  # 使用均值进行填充
  imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
  X_missing_mean = imp_mean.fit_transform(X_missing)
  
  # 使用0进行填充
  imp_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
  X_missing_0 = imp_0.fit_transform(X_missing)
  ```

* 使用随机森林填补缺失值

  任何回归都是从特征矩阵中学习，然后求解连续型标签y的过程，之所以能够实现这个过程，是因为回归算法认为，特征 矩阵和标签之前存在着某种联系。

  实际上，标签和特征是可以相互转换的，比如说，在一个“用地区，环境，附近学校数 量”预测“房价”的问题中，我们既可以用“地区”，“环境”，“附近学校数量”的数据来预测“房价”，也可以反过来， 用“环境”，“附近学校数量”和“房价”来预测“地区”。

  而回归填补缺失值，正是利用了这种思想。 对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他的n-1个特征和原本的标签组成新 的特征矩阵。那对于T来说，它没有缺失的部分，就是我们的Y_test，这部分数据既有标签也有特征，而它缺失的部 分，只有特征没有标签，就是我们需要预测的部分。 特征T不缺失的值对应的其他n-1个特征 + 本来的标签：X_train 特征T不缺失的值：Y_train 特征T缺失的值对应的其他n-1个特征 + 本来的标签：X_test 特征T缺失的值：未知，我们需要预测的Y_test 这种做法，对于某一个特征大量缺失，其他特征却很完整的情况，非常适用。 

  那如果数据中除了特征T之外，其他特征也有缺失值怎么办？

   答案是`遍历所有的特征，从缺失最少的开始进行填补`（因为填补缺失最少的特征所需要的准确信息最少）。 填补一个特征时，先将其他特征的缺失值用0代替，每完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填 补下一个特征。每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，需要用0来填补的特征就越来越少。当 进行到最后一个特征时（这个特征应该是所有特征中缺失值最多的），已经没有任何的其他特征需要用0来进行填补了， 而我们已经使用回归为其他特征填补了大量有效信息，可以用来填补缺失最多的特征。 遍历所有的特征后，数据就完整，不再有缺失值了。

  ``` python
  # 使用均值进行填充
  imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
  X_missing_mean = imp_mean.fit_transform(X_missing)
  
  # 使用0进行填充
  imp_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
  X_missing_0 = imp_0.fit_transform(X_missing)
  ```

* 使用随机森林填补缺失值

  ```python
  """使用随机森林填补缺失值"""
  X_missing_reg = X_missing.copy()
  # 按缺失值的多少进行排序，返回排序后的特征的索引
  sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
  
  
  for i in sortindex:#从缺失值最少的那个特征进行填充
      # 构建新特征矩阵和新标签
      df = X_missing_reg
      fillc = df.iloc[:,i] # 第i列是当前需要填充的列
      df = pd.concat([df.iloc[:,df.columns !=i], pd.DataFrame(y_full)]
                     , axis=1)
      # 在新特征矩阵中，对含有缺失值的列，进行0的填补,否则会有很多的nan值存在
      df_0 = SimpleImputer(missing_values=np.nan
                          , strategy='constant'
                          ,fill_value=0).fit_transform(df)
      # 找出训练集和测试集
      y_train = fillc[fillc.notnull()]
      y_test = fillc[fillc.isnull()]
      X_train = df_0[y_train.index, :]
      X_test = df_0[y_test.index, :]
      
      # 用随机森林回归来填补缺失值
      rfc = RandomForestRegressor(n_estimators=100)
      rfc = rfc.fit(X_train, y_train)
      y_predict = rfc.predict(X_test)
      
      #将填补好的特征返回到我们的原始的特征矩阵中
      X_missing_reg.loc[X_missing_reg.iloc[:,i].isnull(),i] = y_predict
    
  ```

* 对填补号的数据进行建模

  ```pytthon
    
  """对填补号的数据进行建模"""
  # 对所有数据进行建模，取得MSE结果
  X = [X_full,X_missing_mean,X_missing_0,X_missing_reg]
  
  mse = []
  std = []
  for x in tqdm(X):
      estimator = RandomForestRegressor(random_state=0,n_estimators=100)
      scores = cross_val_score(estimator
                               ,x
                               ,y_full
                               ,scoring='neg_mean_squared_error'
                               ,cv=10).mean()
      mse.append(scores * -1)
      
  ```

* 用所得的结果画出条形图

  ```python
  """用所得的结果画出条形图"""
  x_labels = ['Full data',
              'Zero Imputation',
              'Mean Imputation',
              'Regressor Imputation']
  colors = ['r', 'g', 'b', 'orange']
  plt.figure(figsize=(12, 8),dpi=500)
  ax = plt.subplot(111)
  for i in np.arange(len(mse)):
      ax.barh(i, mse[i],color=colors[i], alpha=0.6, align='center')
  ax.set_title('Imputation Techniques with Boston Data')
  ax.set_xlim(left=np.min(mse) * 0.9,
               right=np.max(mse) * 1.1)
  ax.set_yticks(np.arange(len(mse)))
  ax.set_xlabel('MSE')
  ax.set_yticklabels(x_labels)
  plt.show()
  ```

  ​	![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/缺失值填充方法对比.png)

#### 全部代码

* 随机森林回归填补缺失值

  ```python
  """导入需要的库"""
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.datasets import load_boston
  from sklearn.impute import SimpleImputer
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import cross_val_score
  from tqdm import tqdm
  
  
  """以波士顿房价数据集为例子，导入完整的数据集并探索"""
  datasets = load_boston()
  datasets.data.shape #总共506*13=6578个数据
  
  X_full,y_full = datasets.data,datasets.target # 完整数据集
  n_samples = X_full.shape[0] # 样本量
  n_features = X_full.shape[1] # 特征量
  
  """为完整数据集放入缺失值"""
  #首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那总共就要有3289个数据缺失
  rng = np.random.RandomState(0)
  missing_rate = 0.5
  n_missing_samples = int(np.floor(n_samples * n_features * missing_rate)) # 3289
  #np.floor向下取整，返回.0格式的浮点数,所以需要将其转换成整数
  
  # 创建0到n_features（这里是13）之间，长度为n_missing_samples（这里是3289）的数组,用来当做列索引
  missing_features = rng.randint(0,n_features,n_missing_samples)
  # 创建0到n_samples（这里是506）之间，长度为n_missing_samples（这里是3289）的数组,用来当做行索引
  missing_samples = rng.randint(0,n_samples,n_missing_samples)
  
  #我们现在采样了3289个数据，远远超过我们的样本量506，所以我们使用随机抽取的函数randint。但如果我们需要的数据量小于我们的样本量506，那我们可以采用np.random.choice来抽样，choice会随机抽取不重复的随机数，因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中
  
  X_missing = X_full.copy()
  y_missing = y_full.copy()
  
  # 将随机抽取的行列，置位Nan
  X_missing[missing_samples,missing_features] = np.nan
  #转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度快到拯救人生，但是在索引等功能上却不如pandas来得好用
  X_missing = pd.DataFrame(X_missing)
  
  """使用0和均值填补缺失值"""
  # 使用均值进行填充
  imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
  X_missing_mean = imp_mean.fit_transform(X_missing)
  
  # 使用0进行填充
  imp_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
  X_missing_0 = imp_0.fit_transform(X_missing)
  
  """使用随机森林填补缺失值"""
  X_missing_reg = X_missing.copy()
  # 按缺失值的多少进行排序，返回排序后的特征的索引
  sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
  
  
  for i in sortindex:#从缺失值最少的那个特征进行填充
      # 构建新特征矩阵和新标签
      df = X_missing_reg
      fillc = df.iloc[:,i] # 第i列是当前需要填充的列
      df = pd.concat([df.iloc[:,df.columns !=i], pd.DataFrame(y_full)]
                     , axis=1)
      # 在新特征矩阵中，对含有缺失值的列，进行0的填补,否则会有很多的nan值存在
      df_0 = SimpleImputer(missing_values=np.nan
                          , strategy='constant'
                          ,fill_value=0).fit_transform(df)
      # 找出训练集和测试集
      y_train = fillc[fillc.notnull()]
      y_test = fillc[fillc.isnull()]
      X_train = df_0[y_train.index, :]
      X_test = df_0[y_test.index, :]
      
      # 用随机森林回归来填补缺失值
      rfc = RandomForestRegressor(n_estimators=100)
      rfc = rfc.fit(X_train, y_train)
      y_predict = rfc.predict(X_test)
      
      #将填补好的特征返回到我们的原始的特征矩阵中
      X_missing_reg.loc[X_missing_reg.iloc[:,i].isnull(),i] = y_predict
      
  """对填补号的数据进行建模"""
  # 对所有数据进行建模，取得MSE结果
  X = [X_full,X_missing_mean,X_missing_0,X_missing_reg]
  
  mse = []
  std = []
  for x in tqdm(X):
      estimator = RandomForestRegressor(random_state=0,n_estimators=100)
      scores = cross_val_score(estimator
                               ,x
                               ,y_full
                               ,scoring='neg_mean_squared_error'
                               ,cv=10).mean()
      mse.append(scores * -1)
      
  """用所得的结果画出条形图"""
  x_labels = ['Full data',
              'Zero Imputation',
              'Mean Imputation',
              'Regressor Imputation']
  colors = ['r', 'g', 'b', 'orange']
  plt.figure(figsize=(12, 8),dpi=500)
  ax = plt.subplot(111)
  for i in np.arange(len(mse)):
      ax.barh(i, mse[i],color=colors[i], alpha=0.6, align='center')
  ax.set_title('Imputation Techniques with Boston Data')
  ax.set_xlim(left=np.min(mse) * 0.9,
               right=np.max(mse) * 1.1)
  ax.set_yticks(np.arange(len(mse)))
  ax.set_xlabel('MSE')
  ax.set_yticklabels(x_labels)
  plt.show()
  ```

---

### 随机森林在乳腺癌数据上的调参

* 在乳腺癌数据上进行一次随 机森林的调参。乳腺癌数据是sklearn自带的分类数据之一。

#### 代码分解

* 导入需要的库

  ```python
  from sklearn.datasets import load_breast_cancer
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import cross_val_score
  from matplotlib import pyplot as plt
  import pandas as pd
  import numpy as np
  from tqdm import notebook
  
  plt.rcParams['font.sans-serif'] = ['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.rcParams["font.family"] = 'Arial Unicode MS'
  ```

* 导入数据集，探索数据

  ```python
  data = load_breast_cancer()
  print('x shape:',data.data.shape)
  print('y shape:',data.target.shape)
  
  # 乳腺癌数据集有569条记录，30个特征，维度不算太高，但是样本量非常烧，过拟合情况可能存在
  """输出"""
  x shape: (569, 30)
  y shape: (569,)
  ```

* 进行一次简单建模，看看模型本身在数据集上的效果

  ```python
  rfc = RandomForestClassifier(n_estimators=100
                               ,random_state=90)
  score_pre = cross_val_score(rfc
                              ,data.data
                              ,data.target
                              ,cv=10).mean()
  score_pre
  ##这里可以看到，随机森林在乳腺癌数据上的表现本就还不错，在现实数据集上，
  ##基本上不可能什么都不调就看到95%以 上的准确率
  
  
  """输出"""
  0.9596491228070174
  ```

* 随机森林调整的第一步：无论如何先来调`n_estimators`

  在这里我们选择学习曲线，可以使用网格搜索吗?可以，但是只有学习曲线，才能看见趋势 倾向是，要看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息 第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何 引起模型整体准确率的变化

  ```python
  score_list = []
  for i in notebook.tqdm(range(0,200,10)):
      rfc = RandomForestClassifier(n_estimators=i+1
                                  ,n_jobs=-1
                                  ,random_state=25)
      score = cross_val_score(rfc
                             ,data.data
                             ,data.target
                             ,cv=10).mean()
      score_list.append(score)
  print('最大精度：',max(score_list))
  print('最大精度所对应的n_estimators:',(score_list.index(max(score_list))*10)+1)
  
  plt.figure(figsize=(12,8))
  plt.plot(range(1,201,10),score_list)
  plt.xlabel('n_estimators')
  plt.ylabel('RF accuracy')
  plt.show()
  
  """输出"""
  最大精度： 0.9631578947368421
  最大精度所对应的n_estimators: 171
  ```

  ​	![20211223IEg7nb](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 23 IEg7nb.jpg)



* 在确定好的范围内（上面最大是171这里取160到180），进一步细化学习

  ```python
  score_list = []
  for i in notebook.tqdm(range(160,181)):
      rfc = RandomForestClassifier(n_estimators= i
                                  ,n_jobs=-1
                                  ,random_state=25)
      score = cross_val_score(rfc
                             ,data.data
                             ,data.target
                             ,cv=10).mean()
      score_list.append(score)
  
  print('最大精度：',max(score_list))
  print('最大精度所对应的n_estimators:',([*range(160,181)][score_list.index(max(score_list))]))
  plt.figure(figsize=(12,8))
  plt.plot(range(160,181),score_list)
  plt.xlabel('n_estimators')
  plt.ylabel('RF accuracy')
  plt.show()
  
  """输出"""
  最大精度： 0.9631578947368421
  最大精度所对应的n_estimators: 163
  ```

  ​	![202112230I6fWe](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 23 0I6fWe.jpg)

  调整n_estimators的效果显著，模型的准确率立刻上升了0.0035。接下来就进入网格搜索，我们将使用网格搜索对 参数一个个进行调整。为什么我们不同时调整多个参数呢?原因有两个:

  1. 同时调整多个参数会运行非常缓慢,耗时，这里只做演示，所以没有花太多的时间
  2. 同时调整多个参数，会让我们无法理解参数的组合是怎么得来的，所以即便 网格搜索调出来的结果不好，我们也不知道从哪里去改。在这里，为了使用复杂度-泛化误差方法(方差-偏差方 法)，我们对参数进行一个个地调整。

* 为网格搜索做准备，书写网格搜索的参数

  有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势
  从曲线跑出的结果中选取一个更小的区间，再跑曲线

  ```python
  param_grid = {'n_estimators':np.arange(0, 200, 10)}
  param_grid = {'max_depth':np.arange(1, 20, 1)}
  param_grid = {'max_leaf_nodes':np.arange(25,50,1)}
  ```

  对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围

  

  有一些参数是可以找到一个范围的，或者说我们知道他们的取值和随着他们的取值，模型的整体准确率会如何变化，这样的参数我们就可以直接跑网格搜索

  ```python
  param_grid = {'criterion':['gini', 'entropy']}
  param_grid = {'min_samples_split':np.arange(2, 2+20, 1)}
  param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)}
  param_grid = {'max_features':np.arange(5,30,1)}
  ```

* 开始按照参数对模型整体准确率的影响程度进行调参，首先调整`max_depth`

  一般根据数据的大小来进行一个试探，乳腺癌数据很小，特征数就30个，所以可以采用1~10，或者1~20这样的试探

  其实更应该画出学习曲线，来观察深度对模型的影响

  ```python
  # 调整max_depth
  
  param_grid = {'max_depth':np.arange(1,20,1)}
  
  rfc = RandomForestClassifier(n_estimators=163
                              ,random_state=25
                              )
  GS = GridSearchCV(rfc,param_grid
                    ,cv=10
                   )
  GS.fit(data.data,data.target)
  print(f"max_depth 最佳参数:{GS.best_params_}")
  print(f"max_depth 最佳精度:{GS.best_score_}")
  
  max_depth 最佳参数:{'max_depth': 8}
  max_depth 最佳精度:0.9631578947368421
  ```

  在这里，我们注意到，将max_depth设置为有限之后，模型的准确率没有变，或者说是下降了。限制max_depth，是为了让模型变得简 单，把模型向左推，而模型整体的准确率下降了，即整体的泛化误差上升了，这说明模型现在位于图像左边，即泛 化误差最低点的左边(偏差为主导的一边)。通常来说，随机森林应该在泛化误差最低点的右边，树模型应该倾向 于过拟合，而不是拟合不足。这和数据集本身有关，但也有可能是我们调整的n_estimators对于数据集来说太大， 因此将模型拉到泛化误差最低点去了。然而，既然我们追求最低泛化误差，那我们就保留这个n_estimators，除非 有其他的因素，可以帮助我们达到更高的准确率。当模型位于图像左边时，我们需要的是增加模型复杂度(增加方差，减少偏差)的选项，因此max_depth应该尽量 大，min_samples_leaf和min_samples_split都应该尽量小。这几乎是在说明，除了max_features，我们没有任何 参数可以调整了，因为max_depth，min_samples_leaf和min_samples_split是剪枝参数，是减小复杂度的参数。 在这里，我们可以预言，我们已经非常接近模型的上限，模型很可能没有办法再进步了。
  那我们这就来调整一下max_features，看看模型如何变化。

* 调整`max_features`

  max_features是唯一一个即能够将模型往左(低方差高偏差)推，也能够将模型往右(高方差低偏差)推的参数。我 们需要根据调参前，模型所在的位置(在泛化误差最低点的左边还是右边)来决定我们要将max_features往哪边调。现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征 越多，模型才会越复杂。max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的最小值。

  ```python
  param_grid = {'max_features':np.arange(5,30,1)}
  
  rfc = RandomForestClassifier(n_estimators=163
                              ,random_state=25)
  GS = GridSearchCV(rfc
                    ,param_grid
                    ,cv=10
                   )
  GS.fit(data.data,data.target)
  print(f"max_depth 最佳参数:{GS.best_params_}")
  print(f"max_depth 最佳精度:{GS.best_score_}")
  
  
  """输出"""
  max_depth 最佳参数:{'max_features': 5}
  max_depth 最佳精度:0.9631578947368421
  ```

  网格搜索返回了max_features的最小值，可见max_features升高之后，模型的准确率降低了。这说明，我们把模 型往右推，模型的泛化误差增加了。前面用max_depth往左推，现在用max_features往右推，泛化误差都增加， 这说明模型本身已经处于泛化误差最低点，已经达到了模型的预测上限，没有参数可以左右的部分了。剩下的那些 误差，是噪声决定的，已经没有方差和偏差的舞台了。如果是现实案例，我们到这一步其实就可以停下了，因为复杂度和泛化误差的关系已经告诉我们，模型不能再进步\n了。
  调参和训练模型都需要很长的时间，明知道模型不能进步了还继续调整，不是一个有效率的做法。
  如果我们希望模型更进一步，我们会选择更换算法，或者更换做数据预处理的方式。
  但是在课上，出于练习和探索的目的，我们继续调整我们的参数，让大家观察一下模型的变化，看看我们预测得是否正确。依然按照参数对模型整体准确率的影响程度进行调参。

* 调整`min_samples_leaf`

  ```python
   #调整min_samples_leaf 
  param_grid={'min_samples_leaf':np.arange(1, 1+10, 1)}
  #对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20 
  #面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围 
  #如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
  rfc = RandomForestClassifier(n_estimators=163
                               ,random_state=25
                              )
  GS = GridSearchCV(rfc,param_grid,cv=10)
  GS.fit(data.data,data.target)
  print(f"min_samples_leaf 最佳参数:{GS.best_params_}")
  print(f"min_samples_leaf 最佳精度:{GS.best_score_}")
  
  """输出"""
  min_samples_leaf 最佳参数:{'min_samples_leaf': 1}
  min_samples_leaf 最佳精度:0.9631578947368421
  ```

  可以看见，网格搜索返回了min_samples_leaf的最小值，并且模型整体的准确率还不变，这和max_depth的情 况一致，参数把模型向左推，但是模型的泛化误差上升了。在这种情况下，我们显然是不要把这个参数设置起来 的，就让它默认就好了。

* 继续尝试`min_samples_split`

  ```python
  param_grid = {'min_samples_split':np.arange(2,2+20,1)}
  
  rfc = RandomForestClassifier(n_estimators=163
                               ,random_state=25
                              )
  GS = GridSearchCV(rfc,param_grid,cv=10)
  GS.fit(data.data,data.target)
  GS.best_params_
  GS.best_score_
  
  print(f"min_samples_split 最佳参数:{GS.best_params_}")
  print(f"min_samples_split 最佳精度:{GS.best_score_}")
  
  """输出"""
  min_samples_split 最佳参数:{'min_samples_split': 2}
  min_samples_split 最佳精度:0.9631578947368421
  ```

  和`min_samples_leaf`一样的结果，返回最小值并且模型整体的准确率未变，所以这个参数最好也默认就行了

* 最后尝试一下`criterion`

  ```python
   #调整Criterion
  param_grid = {'criterion':['gini', 'entropy']}
  rfc = RandomForestClassifier(n_estimators=163
                               ,random_state=25
                              )
  GS = GridSearchCV(rfc,param_grid,cv=10)
  GS.fit(data.data,data.target)
  print(f"criterion 最佳参数:{GS.best_params_}")
  print(f"criterion 最佳精度:{GS.best_score_}")
  
  """输出"""
  criterion 最佳参数:{'criterion': 'entropy'}
  criterion 最佳精度:0.9666353383458647
  ```

  随机森林默认的是用‘gini’系数，这里entropy使模型结果上升了，有时候就是这样，有时候调整criterion一点用都没有，有时候如神来之笔

* 调整完毕，总结出模型的最佳参数

  ```python
  rfc = RandomForestClassifier(criterion='entropy'
                               ,n_estimators=163
                               ,random_state=25)
  score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
  print('调参后最佳精度：',score)
  print('提升',score - score_pre)
  
  """输出"""
  调参后最佳精度： 0.9666353383458647
  提升 0.006986215538847262
  ```

#### 总结

* 在整个调参过程之中，我们首先调整了`n_estimators`(无论如何都请先走这一步)，然后调整`max_depth`，通过 `max_depth`产生的结果，来判断模型位于复杂度-泛化误差图像的哪一边，从而选择我们应该调整的参数和调参的 方向。如果感到困惑，也可以画很多学习曲线来观察参数会如何影响我们的准确率，选取学习曲线中单调的部分来 放大研究(如同我们对n_estimators做的)。学习曲线的拐点也许就是我们一直在追求的，最佳复杂度对应的泛化 误差最低点，（也就是方差和偏差的平衡点） 网格搜索也可以一起调整多个参数，但是需要大量的时间，有时 候，它的结果比我们的好，有时候，我们手动调整的结果会比较好。当然了，乳腺癌数据集非常完美，所以 只需要调`n_estimators`和`criterion`这两个参数（仅仅在random_state=25的情况下）就达到了随机森林在这个数据集上表现得极限。

---

# 三、数据预处理和特征工程

## 概述

### 故事

* 来自《数据挖掘导论》的一篇故事

  某一天 你从你的同事，一位药物研究人员那里，得到了一份病人临床表现的数据。药物研究人员用前四列数据预测一下最 后一数据，还说他要出差几天，可能没办法和你一起研究数据了，希望出差回来以后，可以有个初步分析结果。于 是你就看了看数据，看着很普通，预测连续型变量，好说，导随机森林回归器调出来，调参调呀调，MSE很小，跑 了个还不错的结果
  $$
  012\quad232\quad33.5\quad0\quad107\quad\\
  020\quad121\quad16.9\quad2\quad210.1\\
  027\quad165\quad24.0\quad0\quad427.6
  $$
  几天后，你同事出差回来了，准备要一起开会了，会上你碰见了和你同事在同一个项目里工作的统计学家。他问起 你的分析结果，你说你已经小有成效了，统计学家很吃惊，他说：“不错呀，这组数据问题太多，我都分析不出什 么来。” 

  你心里可能咯噔一下，忐忑地回答说：“我没听说数据有什么问题呀。” 

  统计学家：“第四列数据很坑爹，这个特征的取值范围是1~10，0是表示缺失值的。而且他们输入数据的时候出错， 很多10都被录入成0了，现在分不出来了。”

  你：”......“ 

  统计学家：”还有第二列和第三列数据基本是一样的，相关性太强了。

  “ 你：”这个我发现了，不过这两个特征在预测中的重要性都不高，无论其他特征怎样出错，我这边结果里显示第一 列的特征是最重要的，所以也无所谓啦。

  “ 统计学家：“啥？第一列不就是编号吗？”

   你：“不是吧。”

  统计学家：“哦我想起来了！第一列就是编号，不过那个编号是我们根据第五列排序之后编上去的！这个第一列和 第五列是由很强的联系，但是毫无意义啊！” 

  你老血喷了一屏幕，数据挖掘工程师卒。

### 数据挖掘的流程

* 数据不给力，再高级的算法都没用，前面两张用的数据，都是sklearn中自带的，都是经过层层筛选适合于算法案例---运行时间段，预测效果好，没有严重缺失等问题。sklearn中的数据，堪称完美。各大机器学习教材也是如此，都给大家提供处理好的数据， 这就导致，很多人在学了很多算法之后，到了现实应用之中，发现模型经常就调不动了，因为现实中的数据，离平 时使用的完美案例数据集，相差十万八千里。

* `数据挖掘的流程`：

  | 数据挖掘的五大流程：                                         |
  | ------------------------------------------------------------ |
  | 1. `获取数据` <br /><br />2. `数据预处理` <br />数据预处理是从数据中检测，纠正或删除损坏，不准确或不适用于模型的记录的过程 <br />**可能面对的问题有**：数据类型不同，比如有的是文字，有的是数字，有的含时间序列，有的连续，有的间断。 也可能，数据的质量不行，有噪声，有异常，有缺失，数据出错，量纲不一，有重复，数据是偏态，数据量太 大或太小 <br />**数据预处理的目的**：让数据适应模型，匹配模型的需求 <br /><br />3. `特征工程`：<br /> 特征工程是将原始数据转换为更能代表预测模型的潜在问题的特征的过程，可以通过挑选最相关的特征，提取 特征以及创造特征来实现。其中创造特征又经常以降维算法的方式实现。<br /> **可能面对的问题有**：特征之间有相关性，特征和标签无关，特征太多或太小，或者干脆就无法表现出应有的数 据现象或无法展示数据的真实面貌 <br />**特征工程的目的**：1) 降低计算成本，2) 提升模型上限 <br /><br />4. `建模`:<br />测试模型并预测出结果 <br /><br />5. `上线`，验证模型效果 |

### sklearn中数据预处理和特征工程

* sklearn中包含众多数据预处理和特征工程相关的模块，虽然刚接触sklearn时，大家都会为其中包含的各种算法的 广度深度所震惊，但其实sklearn六大板块中有两块都是关于数据预处理和特征工程的，两个板块互相交互，为建 模之前的全部工程打下基础。

  ​	![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211223222708.png)

  1. 模块`preprocessing`：几乎包含数据预处理的所有内容 
  2. 模块`Impute`：填补缺失值专用 
  3. 模块`feature_selection`：包含特征选择的各种方法的实践模块
  4. 模块`decomposition`：包含降维算法

---

## 数据预处理

### 数据无量纲化

* 在机器学习算法实践中，我们往往有着将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布 的需求，这种需求统称为将数据“无量纲化”。譬如梯度和矩阵为核心的算法中，譬如逻辑回归，支持向量机，神经 网络，无量纲化可以加快求解速度；而在距离类模型，譬如K近邻，K-Means聚类中，无量纲化可以帮我们提升模 型精度，避免某一个取值范围特别大的特征对距离计算造成影响。（一个特例是决策树和树的集成算法们，对决策 树我们不需要无量纲化，决策树可以把任意数据都处理得很好。）
* 数据的无量纲化可以是线性的，也可以是非线性的。线性的无量纲化包括中心化（Zero-centered或者Meansubtraction）处理和缩放处理（Scale）。中心化的本质是让所有记录减去一个固定值，即让数据样本数据平移到 某个位置。缩放的本质是通过除以一个固定值，将数据固定在某个范围之中，取对数也算是一种缩放处理。

#### 数据归一化

* 对于`StandardScaler`和`MinMaxScaler`来说，空值`NaN`会**被当做是缺失值**，在fit的时候忽略，在transform的时候 保持缺失NaN的状态显示。并且，尽管去量纲化过程不是具体的算法，但在fit接口中，依然只允许导入至少二维数 组，一维数组导入会报错。通常来说，我们输入的X会是我们的特征矩阵，现实案例中特征矩阵不太可能是一维所 以不会存在这个问题。

##### preprocessing.MinMaxScaler

* 当数据 $x$ 按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到 [0,1]之间，而这个过程，就叫做数据归一化(`Normalization`，又称`Min-Max Scaling`)。注意，Normalization是归 一化，不是正则化，真正的正则化是regularization，不是数据预处理的一种手段。归一化之后的数据服从正态分 布，公式如下：
  $$
  x^* = \frac{x-min(x)}{max(x)-min(x)}
  $$

* 在sklearn当中，我们使用preprocessing.MinMaxScaler来实现这个功能。MinMaxScaler有一个重要参数， feature_range，控制我们希望把数据压缩到的范围，默认是[0,1]

  ```python
  from sklearn.preprocessing import MinMaxScaler
  import pandas as pd
  
  data =  [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
  data = pd.DataFrame(data,columns=["x",'y']) # 数据
  
  
  # 实现归一化
  min_max_scaler = MinMaxScaler()
  result = min_max_scaler.fit_transform(data) # 归一化后的结果
  
  returned_data = min_max_scaler.inverse_transform(result) # 将归一化后的结果逆转(还原)
  
  
  
  #当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
  #此时使用partial_fit作为训练接口
  #scaler = scaler.partial_fit(data)
  
  
  #使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
  min_max_scaler = MinMaxScaler(feature_range=[5,10]) #依然实例化
  result_1 = min_max_scaler.fit_transform(data) # 归一化到5-10区间的结果
  ```

  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211223224717.png)

  

  

##### numpy归一化

* 使用`numpy`来实现数据归一化

  ```python
  import numpy as np
  import pandas as pd
  
  data =  [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
  data = pd.DataFrame(data,columns=["x",'y'])
  X = data.values
  
  # 归一化
  X_min_max = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) # 归一化后的结果
  
  
  #逆转归一化
  X_returned = X_min_max * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0) # 还原
  ```

  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211223224659.png)

  

  

---

#### 数据标准化

##### preprocessing.StandardScaler

* 当数据 $x$ 按均值($μ$)中心化后，再按标准差($σ$)缩放，数据就会服从为均值为0，方差为1的分布，而这个过程，就叫做数据标准化(`Standardization`，又称`Z-score normalization`)，公式如下：
  $$
  x^* = \frac{x-\mu}{\sigma}
  $$

  ```python
  from sklearn.preprocessing import StandardScaler
  import pandas as pd
  
  data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
  data = pd.DataFrame(data) # 数据
  
  z_score_scaler = StandardScaler()
  result = z_score_scaler.fit_transform(data) # 标准化的结果
  
  returned_data = z_score_scaler.inverse_transform(result) #还原
  ```

  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211223224635.png)

##### numpy标准化

* 使用`numpy`来实现标准化

  ```python
  import pandas as pd
  import numpy as np
  
  data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
  data = pd.DataFrame(data) # 数据
  X = data.values
  
  result = (X-X.mean(axis=0)) / X.std(axis=0) # 标准化
  
  returned_result = result * X.std(axis=0) + X.mean(axis=0) # 还原
  ```

  ![](https://gitee.com/qinchihongye/pic_windows_md/raw/master/20211223224854.png)

---

#### 归一化标准化选哪个

* 看情况。大多数机器学习算法中，会选择StandardScaler来进行特征缩放，因为MinMaxScaler对异常值非常敏 感。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，StandardScaler往往是最好的选择。 
* MinMaxScaler在不涉及距离度量、梯度、协方差计算以及数据需要被压缩到特定区间时使用广泛，比如数字图像 处理中量化像素强度时，都会使用MinMaxScaler将数据压缩于[0,1]区间之中。
*  建议先试试看StandardScaler，效果不好换MinMaxScaler。 除了StandardScaler和MinMaxScaler之外，sklearn中也提供了各种其他缩放处理（中心化只需要一个pandas广 播一下减去某个数就好了，因此sklearn不提供任何中心化功能）。比如，在希望压缩数据，却不影响数据的稀疏 性时（不影响矩阵中取值为0的个数时），我们会使用MaxAbsScaler；在异常值多，噪声非常大时，我们可能会选 用分位数来无量纲化，此时使用RobustScaler。更多详情请参考以下列表。

---

### 缺失值填充

* 机器学习和数据挖掘中所使用的数据，永远不可能是完美的。很多特征，对于分析和建模来说意义非凡，但对于实际收集数据的人却不是如此，因此数据挖掘之中，常常会有重要的字段缺失值很多，但又不能舍弃字段的情况。因此，数据预处理中非常重要的一项就是处理缺失值。

  ```python
  from sklearn.impute import SimpleImputer
  import pandas as pd
  
  data = pd.read_csv('./数据/Narrativedata.csv',engine='python',index_col=0)
  data.head(20)
  ```

  ![20211224inVpGt](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 24 inVpGt.png)

  在这里，我们使用从泰坦尼克号提取出来的样例数据，这个数据有三个特征，一个数值型，两个字符型，标签也是字符 型。

#### impute.SimpleImputer

* `SImpleImputer`类

  ```python
  sklearn.impute.SimpleImputer(missing_values=nan
                               , strategy=’mean’
                               , fill_value=None
                               , verbose=0
                               ,copy=True)
  ```

  | 参数             | 含义&输入                                                    |
  | ---------------- | ------------------------------------------------------------ |
  | `missing_values` | 告诉SimpleImputer，数据中的缺失值长什么样，默认空值np.nan    |
  | `strategy`       | 我们填补缺失值的策略，默认均值。 <br />输入“mean”使用均值填补（仅对数值型特征可用） <br />输入“median"用中值填补（仅对数值型特征可用） <br />输入"most_frequent”用众数填补（对数值型和字符型特征都可用） <br />输入“constant"表示请参考参数“fill_value"中的值（对数值型和字符型特征都可用） |
  | `fill_value`     | 当参数startegy为”constant"的时候可用，可输入字符串或数字表示要填充的值，常用0 |
  | `copy`           | 默认为True，将创建特征矩阵的副本，反之则会将缺失值填补到原本的特征矩阵中去。 |

* 填充缺失值

  ```python
  #填补年龄
  Age = data.loc[:,"Age"].values.reshape(-1,1) #sklearn当中特征矩阵必须是二维
  
  imp_mean = SimpleImputer() #实例化，默认均值填补
  imp_median = SimpleImputer(strategy="median") #用中位数填补
  imp_0 = SimpleImputer(strategy="constant",fill_value=0) #用0填补
  
  imp_mean = imp_mean.fit_transform(Age) #fit_transform一步完成调取结果
  imp_median = imp_median.fit_transform(Age)
  imp_0 = imp_0.fit_transform(Age)
  
  #在这里我们使用中位数填补Age
  data.loc[:,"Age"] = imp_median
  
  
  #使用众数填补Embarked
  # 仓位等级，使用众数填充比较合理
  Embarked = data.loc[:,"Embarked"].values.reshape(-1,1) 
  
  imp_mode = SimpleImputer(strategy = "most_frequent")
  data.loc[:,"Embarked"] = imp_mode.fit_transform(Embarked)
  data.info()
  ```

#### pandas、numpy

* 使用pandas填充

  ```python
  import pandas as pd
  data = pd.read_csv('./数据/Narrativedata.csv',engine='python',index_col=0)
  
  data.head()
  # 中位数填充
  data.loc[:,"Age"] = data.loc[:,"Age"].fillna(data.loc[:,"Age"].median()) # 
  #.fillna 在DataFrame里面直接进行填补
  
  data.dropna(axis=0,inplace=True)
  #.dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1)删除所有有缺失值的列
  #参数inplace，为True表示在原数据集上进行修改，为False表示生成一个复制对象，不修改原数据，默认False
  ```

---

### 处理分类型特征：编码与哑变量

* 在机器学习中，大多数算法，譬如逻辑回归，支持向量机SVM，k近邻算法等都只能够处理数值型数据，不能处理 文字，在sklearn当中，除了专用来处理文字的算法，其他算法在fit的时候全部要求输入数组或矩阵，也不能够导 入文字型数据（其实手写决策树和普斯贝叶斯可以处理文字，但是sklearn中规定必须导入数值型）。 然而在现实中，许多标签和特征在数据收集完毕的时候，都不是以数字来表现的。比如说，学历的取值可以是["小 学"，“初中”，“高中”，"大学"]，付费方式可能包含["支付宝"，“现金”，“微信”]等等。在这种情况下，为了让数据适 应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型。

#### preprocessing.LabelEncoder：

* 标签专用，能够将分类转换为分类数值

  ```python
  data.head(10)
  ```

  ![20211227E6eC0o](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 E6eC0o.png)

  ```python
  from sklearn.preprocessing import LabelEncoder
  y = data.iloc[:,-1] #要输入的是标签，不是特征矩阵，所以允许一维
  le = LabelEncoder() #实例化
  
  label = le.fit_transform(y)  # label_encoder
  ruturned_label = le.inverse_transform(label) #使用inverse_transform可以逆转
  
  data.iloc[:,-1] = label # #让标签等于我们运行出来的结果
  
  # 简便的写法为
  # data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])
  ```

  ![20211227OGCnff](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 OGCnff.png)

#### preprocessing.OneHotEncoder

* 特征专用，能够将分类特征转换为分类数值

  ```python
  from sklearn.preprocessing import OrdinalEncoder
  
  #接口categories_对应LabelEncoder的接口classes_，一模一样的功能
  data_ = data.copy()
  data_.head(10)
  ```

  ![202112270hnpbK](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 0hnpbK.png)

  ```python
  # sex和Embarked两列
  OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_ 
  
  """输出"""
  [array(['female', 'male'], dtype=object), array(['C', 'Q', 'S'], dtype=object)]
  ```

  ```python
  data_.iloc[:,1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
  data_.head()
  ```

  ![20211227bQDVSS](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 bQDVSS.png)

#### preprocessing.OneHotEncoder

* 独热编码，创建哑变量

  我们刚才已经用OrdinalEncoder把分类变量Sex和Embarked都转换成数字对应的类别了。在舱门Embarked这一 列中，我们使用[0,1,2]代表了三个不同的舱门，然而这种转换是正确的吗？

  我们来思考三种不同性质的分类数据： 

  1.  舱门（S，C，Q） 

     三种取值S，C，Q是相互独立的，彼此之间完全没有联系，表达的是S≠C≠Q的概念。这是`名义变量`。 

  2.  学历（小学，初中，高中） 

     三种取值不是完全独立的，我们可以明显看出，在性质上可以有高中>初中>小学这样的联系，学历有高低，但是学 历取值之间却不是可以计算的，我们不能说小学 + 某个取值 = 初中。这是`有序变量`。

  3. 体重（>45kg，>90kg，>135kg） 

     各个取值之间有联系，且是可以互相计算的，比如120kg - 45kg = 90kg，分类之间可以通过数学计算互相转换。这 是`有距变量(标度变量)`。

     

   然而在对特征进行编码的时候，这三种分类数据都会被我们转换为[0,1,2]，这三个数字在算法看来，是连续且可以 计算的，这三个数字相互不等，有大小，并且有着可以相加相乘的联系。所以算法会把舱门，学历这样的分类特 征，都误会成是体重这样的分类特征。这是说，我们把分类转换成数字的时候，**忽略了数字中自带的数学性质，所 以给算法传达了一些不准确的信息**，而这会影响我们的建模。

   类别OrdinalEncoder可以用来处理有序变量，但对于名义变量，我们只有使用哑变量的方式来处理，才能够尽量 向算法传达最准确的信息：

  ![20211227icXVBk](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 icXVBk.png)

  这样的变化，让算法能够彻底领悟，原来三个取值是没有可计算性质的，是“有你就没有我”的不等概念。在我们的 数据中，性别和舱门，都是这样的名义变量。因此我们需要使用独热编码，将两个特征都转换为哑变量。

  ```python
  data.head(10)
  ```

  ![image-20211227121440878](/Users/mengzhichao/Library/Application Support/typora-user-images/image-20211227121440878.png)

  ```python
  from sklearn.preprocessing import OneHotEncoder
  
  X = data[['Sex','Embarked']]
  
  enc = OneHotEncoder(categories='auto').fit(X)
  result = enc.transform(X).toarray()
  result
  
  """输出"""
  array([[0., 1., 0., 0., 1.],
         [1., 0., 1., 0., 0.],
         [1., 0., 0., 0., 1.],
         ...,
         [1., 0., 0., 0., 1.],
         [0., 1., 1., 0., 0.],
         [0., 1., 0., 1., 0.]])
  ```

  ```python
  #依然可以还原
  pd.DataFrame(enc.inverse_transform(result)).head()
  ```

  ![20211227oLeqoV](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 oLeqoV.png)

  ```python
  enc.get_feature_names() #查看生成的特征名称
  
  """输出"""
  array(['x0_female', 'x0_male', 'x1_C', 'x1_Q', 'x1_S'], dtype=object)
  ```

  ```python
  #合并
  newdata = pd.concat([data
                       ,pd.DataFrame(result,columns=enc.get_feature_names())
                      ]
                      ,axis=1)
  newdata.drop(["Sex","Embarked"],axis=1,inplace=True) # 再将转换前的两列删除
  newdata.head(10)
  ```

  ![20211227pSr83N](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 pSr83N.png)

  特征可以做哑变量，标签也可以吗？可以，使用类sklearn.preprocessing.LabelBinarizer可以对做哑变量，许多算 法都可以处理多标签问题（比如说决策树），但是这样的做法在现实中不常见，因此我们在这里就不赘述了。

  | 编码与哑变量    | 功能                           | 重要参数                                                     | 重要属性                                                     | 重要接口                                                     |
  | --------------- | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | .LabelEncoder   | 分类标签编码                   | N/A                                                          | .classes_:查看变迁中酒精有多少类别                           | fit,<br />transform<br />fit_transform<br />inverse_transform |
  | .OrdinalEncoder | 分类特征编码                   | N/A                                                          | .categories_:查看特征中究竟有多少类别                        | fit,<br />transform<br />fit_transform<br />inverse_transform |
  | .OneHotEncoder  | 独热编码，为名义变量创建哑变量 | categories:每个特征都有哪些类别，默认auto表示range算法自己判断，或者可以输入裂变，每个元素都是一个裂变，表示每个特征中的不同类别<br />handle_unknown:当输入了categories，且算法遇见了categories中没有写明的特征或类别时，是否报错。默认error表示请报错，也可以选择ignore便是请无视。如果选择ignore，则未再categories中注明的特征或类别的哑变量会全部显示为0.在逆转（inverse transvorm）中，未知特征或类别会被返回为None | .categories_:查看特征中酒精有多少类别，如果是自己输入的类别那就不需要查看了 | fit,<br />transform<br />fit_transform<br />inverse_transform<br />get_feature_names:查看生成的哑变量的每一列都是什么特征的什么取值 |

#### 数据类型以及常用统计量

* 常用数据类型和对应常用统计量

  | 数据类型   | 数据名称     | 数学含义  | 描述                                                         | 举例                                                   | 可用操作                                                     |
  | ---------- | ------------ | --------- | ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
  | 离散、定性 | 名义         | =、$\neq$ | 不同的名字，是用来告诉我们这两个数据是否相同                 | 邮编、性别、民族                                       | 众数、信息熵、清醒分析或列联表，相关性分析，卡方检验         |
  | 离散、定性 | 有序         | <, >      | 为数据的相对大小提供信息，告诉我们数据的顺序，但数据之间大小的间隔不是具有固定意义的，因此有序变量不能加减 | 材料的硬度、学历                                       | 中位数，分位数，非参数相关分析（等级相关），测量系统分析，符号检验 |
  | 连续、定量 | 有距（标度） | +, -      | 之间的间隔是有固定意义的，可以加减                           | 日期、以摄氏度华氏度为量纲的温度                       | 均值，标准差，皮尔逊相关系数，t和F检验                       |
  | 连续、定量 | 比率         | *, /      | 比变量之间的间隔和比例本身都是有意义的，既可以加减又可以乘除 | 以开尔文为梁刚的温度，货币数量，计数，年龄，长度，电流 | 几何平均，调和平均，百分数，变化量                           |

### 处理连续型特征：二值化与分段

#### sklearn.preprocessing.Binarizer

* 根据阈值将数据二值化（将特征值设置为0或1），用于处理连续型变量。大于阈值的值映射为1，而小于或等于阈 值的值映射为0。默认阈值为0时，特征中所有的正值都映射到1。二值化是对文本计数数据的常见操作，分析人员 可以决定仅考虑某种现象的存在与否。它还可以用作考虑布尔随机变量的估计器的预处理步骤（例如，使用贝叶斯 设置中的伯努利分布建模）。

  ```python
  data_2 = data.copy()
  data_2.head(10)
  ```

  ![20211227CUJeF3](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 CUJeF3.png)

  ```python
  """年龄二值化"""
  from sklearn.preprocessing import Binarizer
  
  X = data_2.Age.values.reshape(-1,1) #类为特征专用，所以不能使用一维数组
  # 小于等于30映射为0，大于30映射为1
  transformer = Binarizer(threshold=30).fit_transform(X)
  data_2['Age_trans'] = transformer
  data_2.head(10)
  ```

  ![20211227GVcXre](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 GVcXre.png)

#### preprocessing.KBinsDiscretizer

* 这是将连续型变量划分为分类变量的类，能够将连续型变量排序后按顺序分箱后编码。总共包含三个重要参数：

  | 参数     | 含义&输入                                                    |
  | -------- | ------------------------------------------------------------ |
  | n_bins   | 每个特征中分箱的个数，默认5，一次会被运用到所有导入的特征    |
  | encode   | 编码的方式，默认“onehot”<br /> "onehot"：做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该 类别的样本表示为1，不含的表示为0 “ordinal”：每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含 有不同整数编码的箱的矩阵 <br />"onehot-dense"：做哑变量，之后返回一个密集数组 |
  | strategy | 用来定义箱宽的方式，默认"quantile" <br />"uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为 (特征.max() - 特征.min())/(n_bins) <br />"quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同 <br />"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同 |

  ```python
  data_3 = data.copy()
  data_3.head()
  ```

  ![20211227zEIRQX](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 zEIRQX.png)

  ```python
  """ordinal编码"""
  
  
  from sklearn.preprocessing import KBinsDiscretizer
  
  X = data_3.Age.values.reshape(-1,1)
  est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
  result = est.fit_transform(X)
  data_3['Age_ordinal'] = result
  
  #查看转换后分的箱：变成了一列中的三箱
  set(est.fit_transform(X).ravel())
  
  """输出"""
  {0.0, 1.0, 2.0}
  ```

  ```python
  """one-hot编码"""
  
  est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
  
  data_3 = pd.concat([data_3,pd.DataFrame(est.fit_transform(X).toarray())],axis=1)
  
  #查看转换后分的箱：变成了哑变量
  est.fit_transform(X).toarray()
  
  """输出"""
  array([[1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         ...,
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 0.]])
  ```

  ```python
  data_3.head()
  
  ```

  ![20211227rz1wMC](https://gitee.com/qinchihongye/picture-hosting/raw/master/uPic/2021 12 27 rz1wMC.png)

---





















































































































































































