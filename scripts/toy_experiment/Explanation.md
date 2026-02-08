# Toy dataset Explanation
这个文件夹存放的是和toy dataset生成相关的内容。而这一份文件存放的是关于how toy dataset are generated的内容（What prior）

## toy_sine
这个其实是为了测试如何在tslib中使用自己的数据集进行的实验。没有什么特殊之处，仅仅是跑通代码

## toy_complex
这也是一个随机的测试，是一个较难的toy dataset

## toy_complicated
这也是一个随机的测试，是一个较难的toy dataset

## toy_iid_test
上述的三个toy experiment所使用的toy dataset是用窗口切分一个长多元时间序列得出来的，但是这貌似不是一个好主意
因此，现在希望数据集是逐一独立生成的。这个数据集里面的数据都是随机噪声，只是用来跑通新的Toy_Dataset class的

## toy_data_attempt1&2
这两个数据集稍微尝试了一下使用directory of prior中的先验指导生成，但是dataset质量估计非常的堪忧。。。

## syn_periodicity.sh
这个是用来验证periodicity建模的实验。对应的数据集是syn_period_generation.py