import pdfplumber as pbr, numpy as np, os,re
import unittest
from io import StringIO as sio
from sklearn.cluster import MeanShift
from tqdm import tqdm

from typing import Any, Callable

class LineTests(object):
    '''
    One or more tests, a single line object
    '''

    def __init__(self,*tests:list[Callable]):
        self.tests = tests
    
    def get_individual_test_results(self,lineobj):
        return [test(lineobj) for test in self.tests]
    
    def test(self, lineobj:dict, func:Callable = lambda arg: None) -> list[bool]:
        out = self.get_individual_test_results(lineobj)
        outf = func(out)
        return outf if outf != None else out
    
class LinesTests(LineTests):
    '''
    One or more tests, one or more line objects
    '''

    def get_individual_test_results(self,lineobjs):
        return [b for test in self.tests for b in test(lineobjs)]

dr = r'Downloads\fnma_cas'
fn  = r'connave-2021-r01-offering-memorandum.pdf'

path_ = os.path.join(dr,fn)  
path = path_ if os.path.isfile(path_) else os.path.join(r'..\..\..',path_)
print(path)

removepagenum = lambda lines: lines[:-1 if pagenumtest.test(lines[-1]) else lines]

pagenumtest = LineTests(lambda line: re.search('\d+',line['text'].strip()))

lastlinepunctest = LineTests(lambda line: line['text'].strip()[-1] in '. ? ! : ,'.split())
lastlineendtest = LineTests(lambda line: line['x1'] < 500)
firstlinestarttest = LineTests(lambda line: line['x0'] > 100)
nextlineisindented = firstlinestarttest

# TODO: write nextline indent test to apply only to commas and possibly colons.
linetests = LinesTests(
    lambda lobjs: lastlinepunctest.test(lobjs[0]),
    lambda lobjs: lastlineendtest.test(lobjs[0]),
    lambda lobjs: nextlineisindented.test(lobjs[1])
    )

########################################

# print(linetests.test((removepagenum(pagelines[0])[-1],pagelines[1][0])))





# class MakeParagraphs(object):

#     @staticmethod
#     def get_dists(lines:list[Any], # typically output from pdfplumber extract_text_lines()
#                   features = 'top bottom'.split(),
#                   ) -> list[float]: return [line2[features[0]] - line1[features[1]] for line1,line2 in zip(lines[:-1],lines[1:])]

#     @staticmethod
#     def test_para_end(line: Any):pass

#     def __init__(pbrobj: Any,test:bool =True):
#         pass
        


#     def proc_prosp(self,path):
#         pbdf = pbr.open(path)
#         pagear = []
#         pages = pbdf.pages
#         for page in tqdm(pages):
#             lines = page.extract_text_lines()
#             dists = np.array(get_dists(lines))
#             if not dists.size: continue
#             ms = MeanShift().fit(dists.reshape(-1, 1))
#             clusters = ms.predict(dists.reshape(-1, 1))
#             nobreak = clusters[np.argmin(dists)]
#             siox = sio()
#             for line,label in zip(lines[:-1],clusters): # distances to next line stop at lines[-2]
#                 siox.write(line['text'])
#                 siox.write('\n') if label != nobreak else siox.write(' ')
#             siox.write(lines[-1]['text'])
#             pagear.append(siox.getvalue())
#             siox.close()
#         return pagear

class TestBase(unittest.TestCase):
    pass

class TestLastFirstLines(TestBase):

    doc = pbr.open(path)

    def setUp(self):
        self.pagelines = [page.extract_text_lines() for page in tqdm(self.doc.pages[126:130])]    
    
    def test_removepagenum(self):
        self.assertNotEqual(removepagenum(self.pagelines[0])[-1]['text'] != self.pagelines[0][-1]['text'], '%s has not been removed' % self.pagelines[0][-1]['text'])
    
    # @unittest.skip('')
    def test_last_first_lines(self):
        self.assertEquals(linetests.test((removepagenum(self.pagelines[0])[-1],self.pagelines[1][0])),[[True],[True]])
        

def suite():
    suite = unittest.TestSuite()
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFileSelection))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDirectorySelection))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopflag))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWalk))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUpdateInitialization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLastFirstLines))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ToyTestPathMethods))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRexs))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtensions))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNames))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopflag))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
#     unittest.main()