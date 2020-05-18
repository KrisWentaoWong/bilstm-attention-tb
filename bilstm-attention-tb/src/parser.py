# -*- coding: utf-8 -*
from optparse import OptionParser
from arc_hybrid import ArcHybridLSTM
from logger import logger
import pickle, utils, os, time, sys, logging

if __name__ == '__main__':
    parser = OptionParser()
    #训练集路径
    # parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="cn_data/ctb8_conllx/train.conllx")
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="en_data/tb/train.conll")
    #发展集路径
    # parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="cn_data/ctb8_conllx/dev.conllx")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="en_data/tb/dev.conll")
    #测试集路径
    # parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="cn_data/ctb8_conllx/test.conllx")    
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="en_data/tb/test.conll")
    #测试集金标准集
    # parser.add_option("--golden", dest="conll_golden", help="Annotated CONLL golden file", metavar="FILE", default="cn_data/ctb8_conllx/test.golden.conllx")
    parser.add_option("--golden", dest="conll_golden", help="Annotated CONLL golden file", metavar="FILE", default="en_data/tb/test.golden.conll")

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid/src/result/barchybrid.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=3)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--k", type="int", dest="window", default=3)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="barchybrid/src/result")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=7)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_option("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--dynet-mem", type="int", dest="cnn_mem", default=1024)

    parser.add_option("--attention", action="store_true", dest="attentionFlag", default=False)
    parser.add_option("--lstm-output-size", type="int", dest="lstm_output_size", default=125)
    parser.add_option("--encoder-output-size", type="int", dest="encoder_output_size", default=1200)

    

    (options, args) = parser.parse_args()
    print '使用外部词向量:', options.external_embedding
    # logger.info('使用外部词向量:%s', options.external_embedding)

    print '激活函数:',options.activation

    print 'lstm层数:',options.lstm_layers

    # 训练模型过程，先在训练集上进行训练，再在发展集上测试
    if not options.predictFlag:
        if not (options.rlFlag or options.rlMostFlag or options.headFlag):
            print '必须加上参数 --userlmost 或 --userl 或 --usehead (可以使用多个))'
            # logger.warning('必须加上参数 --userlmost 或 --userl 或 --usehead (可以使用多个))')
            sys.exit()

        print '准备数据'
        # logger.info('准备数据')
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print '准备数据结束'
        # logger.info('准备数据结束')

        print '初始化 blstm arc hybrid:'
        # logger.info('初始化 blstm arc hybrid:')
        parser = ArcHybridLSTM(words, pos, rels, w2i, options)

        for epoch in xrange(options.epochs):
            print '开始 第', epoch ,'轮'
            # logger.info('开始轮次 '+ str(epoch))
            parser.Train(options.conll_train)
            conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch+1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(devpath, parser.Predict(options.conll_dev))

            # 如果不是conllu格式
            if not conllu:
                os.system('perl barchybrid/src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
            # 是conllu格式
            else:
                os.system('/usr/bin/python barchybrid/src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
            
            print '预测发展集结束'
            # logger.info('预测发展集结束')
            parser.Save(options.model + str(epoch+1))
    # 测试过程，在测试集上
    else:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        for epoch in xrange(options.epochs):
            print '开始 第', epoch ,'轮'
            # 从已经训练好的模型中载入模型
            parser.Load(options.model+str(epoch+1))
            print '模型：'+options.model+str(epoch+1)
            # 根据文件名后缀判断是否是conllu文件
            conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
            tespath = os.path.join(options.output, 'test_pred'+str(epoch+1)+'.conll' if not conllu else 'test_pred'+str(epoch+1)+'.conllu')
            print '测试结果路径：',tespath
            # logger.info('测试结果路径：%s',tespath)
            ts = time.time()
            pred = list(parser.Predict(options.conll_test))
            te = time.time()
            utils.write_conll(tespath, pred)

            print '金标准路径:',options.conll_golden
            # logger.info('金标准路径：%s',options.conll_golden)
            if not conllu:
                os.system('perl barchybrid/src/utils/eval.pl -g ' + options.conll_golden + ' -s ' + tespath  + ' > ' + tespath + '.txt')
            else:
                os.system('/usr/bin/python barchybrid/src/utils/evaluation_script/conll17_ud_eval.py -v -w barchybrid/src/utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + tespath + ' > ' + testpath + '.txt')
            
            print '预测测试集结束，用时：',te-ts,'秒'
            # logger.info('预测测试集结束，用时：%s秒',te-ts)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    