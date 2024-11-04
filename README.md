  在data文件中，Copy of Data和AWM是原始数据集。MergerAWM是AWM中两张数据表的合并。前缀为2的数据集经过了预处理操作，如去除重复值缺失值，文本值转数值。前缀为3的数据集经过了标签编码，例如将67个职业按顺序从0-66编码。
  在encoder文件中是5个json文件，能在python中解析成字典，字典中存储着每一个变量对应的编码规则，例如在Occupation（职业）中0对应着物流职业。
  在model文件中原本存储着模型，但是模型太大github上传不了，就没有上传了。
  labelencode是编码文件，收集数据集中的唯一值，然后按顺序编码，最后用字典保存在本地。
  models文件是建模的，里面有多个模型，全部使用K折交叉验证，可以选择使用标准化或者归一化。
  preprocessing_1是对Copy of Data预处理的文件
  preprocessing_1是对MergerAWM预处理的文件
  savemodel是单独挑出XGB和RF进行建模的文件，同时使用votting投票分类器，看看效果好不好
  statistics是对数据集的数学特征生成，其中包括了相关系数矩阵和scree plot图
  time是对数据的时间权重进行研究，暂时还没弄好
