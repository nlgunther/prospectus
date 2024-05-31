import pdfplumber as pbr, numpy as np, os
from io import StringIO as sio
from sklearn.cluster import MeanShift
from tqdm.notebook import tqdm

dr = r'Downloads\fnma_cas'
fn  = r'connave-2021-r01-offering-memorandum.pdf'

samplepath = os.path.join(dr,fn)
path = samplepath


def proc_prosp(path):
    pbdf = pbr.open(path)

    get_dists = lambda lines: [line2['top'] - line1['bottom'] for line1,line2 in zip(lines[:-1],lines[1:])]

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
        for line,label in zip(lines,clusters):
            siox.write(line['text'])
            siox.write('\n') if label != nobreak else siox.write(' ')
        pagear.append(siox.getvalue())
        siox.close()
    return pagear