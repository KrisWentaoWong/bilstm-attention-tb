# -*- coding: utf-8 -*
from collections import Counter
from logger import logger
import re

#d定义Conll格式文件的结构
class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        # 当前词在句子中的序号，1开始
        self.id = id
        # 当前词语或标点
        self.form = form
        # 如果是数字返回“NUM”，如果是字符返回小写形式
        self.norm = normalize(form)
        # 当前词语的词性（粗粒度）
        self.cpos = cpos.upper()
        # 当前词语的词性（细粒度）
        self.pos = pos.upper()
        # 当前词语的中心词id
        self.parent_id = parent_id
        # 当前词语与中心词的依存关系
        self.relation = relation

        # 当前词语（或标点）的原型或词干，在中文中，此列与form相同
        self.lemma = lemma
        # 句法特征
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = 0
            root.pred_relation = 'rroot'
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)

    #添加依赖关系
    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]

# 判断依存句法树是否projective
# 方法：判断句中相邻的两个单词，如果存在依存关系，则删除依存词保留中心词。
# 如果是projective，则所有依存关系都是以相邻单词形式出现，因为之前的已经被删除。则最后只剩下一个单词
# 问题：如果被删除的单词是剩下未处理单词的中心词怎么办？
# 解答：程序中通过维护一个map记录以当前单词为中心词的依存词个数，优先处理依存词，即以当前单词为中心词的依存词个数为0。
# 并且每次处理之后对中心词的依存词个数-1，对应之前的删除操作。
def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, True):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys())


#读入数据集，统计句子个数，并且丢弃其中的no-projective句子
def read_conll(fh, proj):
    dropped = 0
    read = 0
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1:
                if not proj or isProj([t for t in tokens if isinstance(t, ConllEntry)]):
                    # 如果一个函数定义中包含 yield 表达式，那么该函数是一个生成器函数。
                    # 与普通函数不同，生成器函数被调用后，其函数体内的代码并不会立即执行，而是返回一个生成器（generator-iterator）。
                    # 当返回的生成器调用成员方法时，相应的生成器函数中的代码才会执行。
                    yield tokens
                else:
                    dropped += 1
                read += 1
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7]))
    if len(tokens) > 1:
        yield tokens

    print dropped, '被丢弃的 non-projective 句子。'
    print read, '被读入的句子。'
    logger.info("%s被丢弃的non-projective 句子。%s被读入的句子。",dropped,read)


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

cposTable = {"PRP$": "PRON", "VBG": "VERB", "VBD": "VERB", "VBN": "VERB", ",": ".", "''": ".", "VBP": "VERB", "WDT": "DET", "JJ": "ADJ", "WP": "PRON", "VBZ": "VERB", 
             "DT": "DET", "#": ".", "RP": "PRT", "$": ".", "NN": "NOUN", ")": ".", "(": ".", "FW": "X", "POS": "PRT", ".": ".", "TO": "PRT", "PRP": "PRON", "RB": "ADV", 
             ":": ".", "NNS": "NOUN", "NNP": "NOUN", "``": ".", "WRB": "ADV", "CC": "CONJ", "LS": "X", "PDT": "DET", "RBS": "ADV", "RBR": "ADV", "CD": "NUM", "EX": "DET", 
             "IN": "ADP", "WP$": "PRON", "MD": "VERB", "NNPS": "NOUN", "JJS": "ADJ", "JJR": "ADJ", "SYM": "X", "VB": "VERB", "UH": "X", "ROOT-POS": "ROOT-CPOS", 
             "-LRB-": ".", "-RRB-": "."}
