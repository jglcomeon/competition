import copy
import time
from multiprocessing import Pool, Manager
import pandas as pd
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, prefix):
        # There are 13 integer features and 26 categorical features

        self._cols_category = ['gender', 'age', 'tagid', 'province',
                               'city', 'make', 'model']

        print("loading original data, ......")
        self.df = pd.read_csv("/Users/gl.j/PycharmProjects/baseline/data/训练集/all_train.txt", sep=',', header=None)
        self.df.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model']
        self.train_len = self.df.shape[0]
        test = pd.read_csv("/Users/gl.j/PycharmProjects/baseline/data/测试集/test.txt", sep=',', header=None)
        test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model']
        self.df = pd.concat([self.df, test])
        del self.df['pid']
        del self.df['time']
        self.df['label'].fillna(-1, inplace=True)
        self.df['label'] = self.df['label'].astype('int')
        self.df['gender'].fillna(2, inplace=True)
        self.df['gender'] = self.df['gender'].astype('int').astype('category').cat.codes
        self.df['age'].fillna(0, inplace=True)
        self.df['age'] = self.df['age'].astype('int').astype('category').cat.codes
        self.df['province'] = self.df['province'].astype('category').cat.codes
        self.df['city'] = self.df['city'].astype('category').cat.codes
        self.df['make'] = self.df['make'].astype('category').cat.codes
        self.df['model'] = self.df['model'].astype('category').cat.codes
        print(len(self.df['city'].unique()))
        print(len(self.df['province'].unique()))
        self._build_catval_vocab()
        print("process tagid........")
        self.df['tagid'].fillna("[0]", inplace=True)
        tagid = []
        for i in range(self.df.shape[0]):
            try:
                tagid.extend(eval(self.df.iloc[i]['tagid']))
            except TypeError:
                print(self.df.iloc[i]['tagid'])
                return

        tagid = list(set(tagid))
        print('tagid category has %d', len(tagid))
        tagid.sort()
        # 处理较慢采用多进程分批处理
        p = Pool(8)
        for i in range(1, 9):
            i_1 = (i-1) * 100000
            i_2 = i * 100000
            print("tagid between %d:%d" % (i_1, i_2))
            temp = copy.copy(self.df.iloc[i_1:i_2])
            p.apply_async(self.process_tagid, (tagid, temp, i_1, i_2))

        print('Waiting for all subprocesses done...')

        p.close()
        p.join()

        print("original data[{}] loaded".format(self.df.shape))

        #self._y = self.df['label']
        #self._X = self.df.loc[:, self.df.columns != 'label']
        #self._preproced_df = None
        #self.run()
        #self.split_save(prefix)

    def _build_catval_vocab(self):
        for c in self._cols_category:
            if c != 'tagid':
                self.df[c] = self.df[c].apply(lambda x: "{}:{}".format(x, 1))

    def run(self):

        print("\n============ preprocessing categorical features, ......")
        self._build_catval_vocab()
        print("\n============ categorical features preprocessed")

        self._preproced_df = pd.concat([self._y, self._X],
                                       axis=1)

    def process_tagid(self, tagid, data, i_1, i_2):
        def process(df):
            res = []
            for i in eval(df):
                res.append(str(tagid.index(i))+":1")
            return " ".join(res)
        data['tagid'] = data['tagid'].apply(lambda x: process(x))
        data.to_csv('{}_{}'.format(i_1, i_2))

    def split_save(self, prefix):
        df = self._preproced_df
        train = df[:self.train_len]
        test = df[self.train_len:]
        outfname = '/Users/gl.j/PycharmProjects/baseline/data/训练集/{}_train12.csv'.format(prefix)
        train.to_csv(outfname, sep='\t', index=False)
        print("data[{}] saved to '{}'".format(train.shape, outfname))
        outfname = '/Users/gl.j/PycharmProjects/baseline/data/测试集/{}_test12.csv'.format(prefix)
        test.to_csv(outfname, sep='\t', index=False)
        print("data[{}] saved to '{}'".format(test.shape, outfname))


if __name__ == "__main__":
    start = time.time()
    tqdm.pandas()
    DataPreprocessor("whole")
    print("spend time is %f" % (time.time() - start))
