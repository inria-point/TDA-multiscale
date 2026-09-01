"""Does inflating the cells lower the coarse band on its own, with no text?

Shuffling words inflates a type's cell by 17% while leaving the frame of centres
and singletons untouched (shuffle_geometry.py), and the coarse band falls 13.4%
while the fine one rises 39.6%. Whether that needs any explanation about meaning
is settled by building the same arrangement out of nothing: centres drawn from a
Gaussian, a blob of k points around each, and the blob radius swept through the
values the real texts and the shuffled texts actually have.

The counts are the measured ones -- 94 singletons and 31 cells making up the
L = 201 points, mean multiplicity ~3.4 -- and the centres are drawn in 24
dimensions, which is what qPHD reports for the real centroid cloud. Nothing here
knows about tokens.

The radius 0.40 row is past the point where the blobs merge and the arrangement
stops being "centres plus cells" at all; it is kept only to show where the
picture breaks.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'experiments'); sys.path.insert(0,'scripts')
import config as cfg
from qphd import qphd
from sink_cluster import bands_of
from three_bands import BANDS
from scipy.spatial.distance import pdist
B=list(BANDS)
L=cfg.L_DEFAULT; DC=24            # размерность облака центров
N_SING, N_CELL = 94, 31           # как в реальных текстах
KS=[2,2,2,3,3,3,4,4,5,6]          # кратности, среднее ~3.4

def make(r_target, rng, dim=DC):
    ks=rng.choice(KS,N_CELL)
    need=L-N_SING
    while ks.sum()<need: ks[rng.integers(N_CELL)]+=1
    while ks.sum()>need:
        j=rng.integers(N_CELL)
        if ks[j]>2: ks[j]-=1
    cen=rng.normal(size=(N_SING+N_CELL,dim))
    pts=[cen[:N_SING]]; owner=[-1]*N_SING
    # сперва с единичным радиусом, потом отмасштабируем
    blobs=[rng.normal(size=(k,dim)) for k in ks]
    for j,(c,bl) in enumerate(zip(cen[N_SING:],blobs)):
        pts.append(c+bl); owner+=[j]*len(bl)
    X=np.vstack(pts)
    scale=pdist(X).mean()
    rad=np.mean([np.linalg.norm(b-b.mean(0),axis=1).mean() for b in blobs])
    # подобрать множитель так, чтобы радиус/масштаб = r_target
    f=r_target*scale/rad
    pts=[cen[:N_SING]]
    for c,bl in zip(cen[N_SING:],blobs): pts.append(c+bl*f)
    X=np.vstack(pts)
    sc=pdist(X).mean()
    rad=np.mean([np.linalg.norm((b*f)-(b*f).mean(0),axis=1).mean() for b in blobs])
    return X, rad/sc

rows=[]
for r in [0.15,0.20,0.232,0.271,0.32,0.40]:
    for rep in range(12):
        rng=np.random.default_rng(1000*rep+int(r*1000))
        X,got=make(r,rng)
        df=qphd(X,q_list=cfg.Q_GRID,rng=rng,
                **cfg.qphd_kwargs(L=L,pool=X,replicates=16))
        rows.append(dict(радиус=r,факт=got,**bands_of(df)))
D=pd.DataFrame(rows)
g=D.groupby('радиус')[['факт']+B].mean()
base=g.loc[0.232]
for b in B: g[b+' %']=(g[b]-base[b])/base[b]*100
pd.set_option('display.width',200)
print(f"синтетика: {N_SING} одиночек + {N_CELL} ячеек = {L} точек, "
      f"центры из {DC}-мерного гаусса, 12 повторов\n")
print(g.round(2).to_string())
print("\n% — относительно радиуса 0.232 (реальный текст); 0.271 = перемешанный")
print("\nреальность для сравнения: перемешивание даёт "
      "крупная −13.4%, мелкая +39.6%, средняя −5.4%")
