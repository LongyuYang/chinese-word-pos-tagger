# -*- coding: utf-8 -*-

import math
import wx

tag_list = ['m', 'v', 'n', 'u', 'a', 'w', 't', 'q', 'nt', 'nr', 'Vg', 'k', 'p',
            'f', 'r', 'vn', 'ns', 'c', 's', 'd', 'ad', 'j', 'l', 'an', 'b', 'i',
            'vd', 'z', 'nz', 'Ng', 'Tg', 'y', 'nx', 'Ag', 'o', 'Dg', 'Bg', 'h',
            'Rg', 'vvn', 'e', 'Mg', 'na', 'Yg']
max_tag_number = len(tag_list)

dictionary = []                                   # word_and_tag词典
tag_transition_matrix =  [[0]*max_tag_number for i in range(max_tag_number)]#转移概率矩阵
tag_frequency_list = [0]*max_tag_number           #记录每个标注出现的次数/初始概率分布

class word_and_tag:                               #dictionary词典中的词类
    def __init__(self, word):
        self.word = word                          #词
        self.tag_vector = [0]*max_tag_number      #tag向量，每个数字表示tag的输出概率

#***********************************************************************
#         比较两字符串大小
#***********************************************************************
def strcmp(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    if len1 != len2:
        return False
    for i in range(len1):
        if s1[i] != s2[i]:
            return False
    return True


#**************************************************************************
#         从语料库中初始化词典和转移矩阵
#**************************************************************************
def build_probability(filename):
    tag_frequency_list = [0]*max_tag_number             #记录每个标注出现的次数
    f = open(filename)
    string_list = []
    for line in f:
        string_list.extend(line.split())
    length = len(string_list)
    total_number = (float)(length)
    pre_tag = ''                                       #记录前一个的tag
    for i in range(0, length):
        string = string_list[i]
        word, tag = string.split('/')
        if word[0] == '[':
            word = word[1:]
        if ']' in tag:
            tag = tag[0:-3]
        flag = False
        if tag not in tag_list:
            continue
        for word_tag in dictionary:
            if strcmp(word, word_tag.word):
                word_tag.tag_vector[tag_list.index(tag)] += 1
                flag = True
                break
        if flag == False:
            newWord = word_and_tag(word)
            newWord.tag_vector[tag_list.index(tag)] += 1
            dictionary.append(newWord)
        tag_frequency_list[tag_list.index(tag)] += 1
        if i >= 1:
            tag_transition_matrix[tag_list.index(pre_tag)][tag_list.index(tag)] += 1
        pre_tag = tag
    for i in range(max_tag_number):
        for j in range(max_tag_number):
            if tag_transition_matrix[i][j] != 0 and tag_frequency_list[i] != 0:
                tag_transition_matrix[i][j] = -math.log((float)\
                (tag_transition_matrix[i][j])/(float)(tag_frequency_list[i]))
    for i in dictionary:
        for j in range(max_tag_number):
            if i.tag_vector[j] != 0 and tag_frequency_list[j] != 0:
                i.tag_vector[j] = -math.log((float)(i.tag_vector[j])\
                                /(float)(tag_frequency_list[j]))
    for i in range(max_tag_number):
        if tag_frequency_list[i] != 0:
            tag_frequency_list[i] = -math.log((float)(tag_frequency_list[i])/total_number)
    f.close()

#**************************************************************************
#                      未登录词词性预测
#**************************************************************************
def tag_predict(word):
    tag_vote = [0.0]*max_tag_number
    dict_length = len(dictionary)
    length = len(word)
    for i in range(length):
        if word[i] >= '0' and word[i] <= '9':
            if i == length-1:                        #全是数字，故预测为数词
                return tag_list.index('m')
            if i == length-3:                         #时间词
                if strcmp( word[i+1]+word[i+2], '年'.decode('utf-8').encode('gbk'))== True or\
                strcmp(word[i + 1] + word[i + 2], '月'.decode('utf-8').encode('gbk')) == True or\
                strcmp(word[i + 1] + word[i + 2], '日'.decode('utf-8').encode('gbk')) == True:
                    return tag_list.index('t')
    for i in range(0, length-1, 2):                   #1-gram查找
        for j in range(dict_length):
            if strcmp(word[i]+word[i+1], dictionary[j].word) == True:
                vote = []
                for number in dictionary[j].tag_vector:
                    if number > 0: vote.append(1)
                    else:
                        vote.append(0)
                for p in range(max_tag_number):
                    tag_vote[p] = tag_vote[p] + vote[p]
                break
    for i in range(0, length-3, 2):                  #2-gram查找
        for j in range(dict_length):
            gram_2 = word[i]+word[i+1]+word[i+2]+word[i+3]
            if strcmp(gram_2, dictionary[j].word) == True:
                vote = []
                for number in dictionary[j].tag_vector:
                    if number > 0: vote.append(1)
                    else:
                        vote.append(0)
                for p in range(max_tag_number):
                    tag_vote[p] = tag_vote[p] + vote[p]
                break
    return tag_vote.index(max(tag_vote))

#**************************************************************************
#                      维特比算法
#**************************************************************************
def Viterbi(filename):
    f = open(filename)
    s = []
    for line in f:
        s.extend(line.split())
    length = len(s)
    cost = [[0]*max_tag_number for i in range(length+1)]
    dict_length = len(dictionary)
    dict_num = 0
    in_dict = False                   #设立标志处理未登录词
    for j in range(dict_length):
        if strcmp(s[length - 1], dictionary[j].word) == True:
            dict_num = j
            in_dict = True
            break
    if in_dict == True:                     #已登录词
        for i in range(max_tag_number):
            if (dictionary[dict_num].tag_vector[i] == 0):
                cost[length][i] = -1
                continue
            cost[length][i] = tag_transition_matrix[i][tag_list.index('w')]\
                            + dictionary[dict_num].tag_vector[i]
    else:                                   #未登录词
        for i in range(max_tag_number):
            cost[length][i] = -1
        predict_index = tag_predict(s[length-1])     #预测未登录词的词性
        cost[length][predict_index] = tag_transition_matrix[predict_index][tag_list.index('w')]
    for i in range(length-1, 0, -1):
        in_dict = False
        for k in range(dict_length):
            if strcmp(s[i-1], dictionary[k].word) == True:
                dict_num = k
                in_dict = True
                break
        if in_dict == True:                   #已登录词
            for j in range(max_tag_number):
                if (dictionary[dict_num].tag_vector[j] == 0):
                    cost[i][j] = -1
                    continue
                cost_list = []
                for p in range(max_tag_number):
                    if cost[i+1][p] > 0:
                        cost_list.append(cost[i+1][p] + tag_transition_matrix[j][p])
                get_min = min(cost_list)
                cost[i][j] = get_min + dictionary[dict_num].tag_vector[j]
        else:                                #未登录词
            for j in range(max_tag_number):
                cost[i][j] = -1
            predict_index = tag_predict(s[i-1])  # 预测未登录词的词性
            cost_list = []
            for p in range(max_tag_number):
                if cost[i + 1][p] > 0:
                    cost_list.append(cost[i + 1][p] + tag_transition_matrix[predict_index][p])
            get_min = min(cost_list)
            cost[i][predict_index] = get_min
    tag_sequence = []
    for i in range(1, length+1):
        min_index = 0
        min_cost = 2147483647
        for j in range(max_tag_number):
            if cost[i][j] > 0 and cost[i][j] < min_cost:
                min_cost = cost[i][j]
                min_index = j
        tag_sequence.append(tag_list[min_index])
    return tag_sequence, s


#**************************************************************************
#          将标注结果保存到文件中
#**************************************************************************
def store_result(file, tag_sequence, s):
    f = open(file, 'w')
    length = len(s)
    str = ''
    for i in range(length):
        is_number = True
        for ch in s[i]:
            if not (ch >= '0' and ch <= '9'):
                is_number = False
                break
        f.write(s[i] + '/' + tag_sequence[i] + ' ')
        str += s[i] + '/' + tag_sequence[i] + '   '
    f.close()
    return str


#**************************************************************************
#         保存词典、转移矩阵方便下次使用
#**************************************************************************
def store_training(file1, file2):
    f = open(file1, 'w')
    for i in dictionary:
        f.write(i.word + ' ')
        for j in range(max_tag_number):
            f.write(str(i.tag_vector[j]) + ' ')
        f.write('\n')
    f.close()
    f = open(file2, 'w')
    for i in range(max_tag_number):
        for j in range(max_tag_number):
            f.write(str(tag_transition_matrix[i][j]) + ' ')
        f.write('\n')
    f.close()


#**************************************************************************
#         从保存的词典、转移矩阵的文件中读取数据
#**************************************************************************
def initiate(file1, file2):
    f = open(file1)
    for line in f:
        s = line.split()
        newWord = word_and_tag(s[0])
        for i in range(max_tag_number):
            newWord.tag_vector[i] = float(s[i+1])
        dictionary.append(newWord)
    f.close()
    f = open(file2)
    line_index = 0
    for line in f:
        s = line.split()
        for j in range(max_tag_number):
            tag_transition_matrix[line_index][j] = (float)(s[j])
        line_index = line_index + 1
    f.close()

#**************************************************************************
#                 建立窗口显示结果
#**************************************************************************
class mywin(wx.Frame):
    def __init__(self, parent, title):
        super(mywin, self).__init__(parent, title = title)
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        stcText = wx.StaticText(panel, id = -1, label = 'Result',  style = wx.ALIGN_CENTER)
        self.text = wx.TextCtrl(panel, style=wx.TE_MULTILINE|wx.TE_READONLY)
        vbox.Add(stcText, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.text, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        font = wx.Font(16, wx.ROMAN, wx.NORMAL, wx.NORMAL)
        self.text.SetFont(font)
        panel.SetSizer(vbox)
        self.SetSize((400, 300))
        self.Center()
        self.Show()
        self.Fit()


#build_probability('199801.txt')
#store_training('dictionary.txt', 'transition_matrix.txt')
initiate('dictionary.txt', 'transition_matrix.txt')
tag_sequence, s = Viterbi('tag_test.txt')
result_str = store_result('result.txt', tag_sequence, s)

app = wx.App()
x = mywin(None, 'Word Tag')
x.text.write(unicode(result_str, 'mbcs'))

app.MainLoop()



