

#TODO 
# 根据model name 将模型一次保存如：tansfrom model: tf1/tf2...;predict model:pm3
# 最终形成类似如下的路径 fc.json(使用FeatureCreate生成)->tf_2->tf_3->pm_4->tf_5->pm_6
# 最终循环该路径使用sklearn pipline结合FeatureCreate 构成预测路径，实现模型自动加载（看情况对于模型类别的定义有无均可）
class ModelDeploy():
    pass