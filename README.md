# bilstm-attention-tb
1. 本文提出了bilstm-tb模型，并简化了特征函数；
2. 本文提出了bilstm-attention-tb模型；
3. bilstm-tb模型和bilstm-attention-tb模型通过参数--attention区分；
4. 训练阶段和测试阶段通过参数--predict区分；
5. 测试阶段需指定模型和参数文件，参数为--model和--params；
6. 训练模型命令示例：

```
python parser.py --dynet-seed 123456789 --dynet-mem 2048 --bibi-lstm --usehead --lstmlayers 3 --hidden2 100 --epochs 10 --k 2
```
7. 测试阶段命令示例：

```
python parser.py --dynet-seed 123456789 --dynet-mem 2048 --bibi-lstm --usehead --lstmlayers 3 --hidden2 100 --epochs 10 --k 2 --predict --model barchybrid/src/result/barchybrid.model --params barchybrid/src/result/params.pickle
```
