import pdfplumber as pbr, numpy as np, os
from io import StringIO as sio
from sklearn.cluster import MeanShift
from tqdm.notebook import tqdm

dr = r'Downloads\fnma_cas'
fn  = r'connave-2021-r01-offering-memorandum.pdf'

samplepath = os.path.join(dr,fn)
path = samplepath


class LineTest(object):

    def __init__(self,lineobj:obj,tests:list[obj]):
        self.lineobj = lineobj
        self.tests = tests
    
    def test(self) -> list[bool]:
        return [test(lineobj) for test in self.tests]


class MakeParagraphs(object):

    @staticmethod
    def get_dists(lines:list(obj),
                  features = 'top bottom'.split(),
                  ) -> list[float]: return [line2[features[0]] - line1[features[1]] for line1,line2 in zip(lines[:-1],lines[1:])]

    @staticmethod
    def test_para_end(line: obj):

    def __init__(pbrobj: obj,test:bool =True):
        


def proc_prosp(path):
    pbdf = pbr.open(path)
    pagear = []
    pages = pbdf.pages
    for page in tqdm(pages):
        lines = page.extract_text_lines()
        dists = np.array(get_dists(lines))
        if not dists.size: continue
        ms = MeanShift().fit(dists.reshape(-1, 1))
        clusters = ms.predict(dists.reshape(-1, 1))
        nobreak = clusters[np.argmin(dists)]
        siox = sio()
        for line,label in zip(lines[:-1],clusters): # distances to next line stop at lines[-2]
            siox.write(line['text'])
            siox.write('\n') if label != nobreak else siox.write(' ')
        siox.write(lines[-1]['text'])
        pagear.append(siox.getvalue())
        siox.close()
    return pagear