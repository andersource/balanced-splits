import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from balanced_splits.split import optimized_split
import io
from PIL import Image

sns.set()

sample_size = 160
max_iter = 5000
df = pd.DataFrame({
    'age': np.random.normal(loc=45, scale=7., size=sample_size),
    'skill': 1 - np.random.power(4, size=sample_size),
    'type': np.random.choice(['T1', 'T2', 'T3'], size=sample_size)
})

iters = [0]
scores = [0]
frames = []
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(ncols=3, nrows=2)
def callback(parts, score):
    iters.append(iters[-1] + 1)
    scores.append(score)

    if iters[-1] not in [1, 10, 50, 100, 200, 500, 1000, 2000, 5000]:
        return

    A, B, C, D = parts

    fig.clf()

    ax = fig.add_subplot(gs[0, 0])
    sns.distplot(A['age'], hist=False, label='A', ax=ax)
    sns.distplot(B['age'], hist=False, label='B', ax=ax)
    sns.distplot(C['age'], hist=False, label='C', ax=ax)
    sns.distplot(D['age'], hist=False, label='D', ax=ax)
    ax.legend()
    ax.set_xlim((18, 82))
    ax.set_ylim((0, 0.075))
    ax.set_title('age')

    ax = fig.add_subplot(gs[0, 1])
    sns.distplot(A['skill'], hist=False, label='A', ax=ax)
    sns.distplot(B['skill'], hist=False, label='B', ax=ax)
    sns.distplot(C['skill'], hist=False, label='C', ax=ax)
    sns.distplot(D['skill'], hist=False, label='D', ax=ax)
    ax.legend()
    ax.set_xlim((0.3, 1.1))
    ax.set_ylim((0, 3.))
    ax.set_title('skill')

    ax = fig.add_subplot(gs[0, 2])
    tmp = pd.DataFrame([{'type': row['type'], 'source': 'ABCD'[i]}
                        for i, p in enumerate([A, B, C, D])
                        for _, row in p.iterrows()])
    sns.countplot(x='type', hue='source', data=tmp, ax=ax,
                  order=['T1', 'T2', 'T3'])
    plt.legend(loc='lower left')
    ax.set_ylim((0, 17))
    ax.set_title('type')

    ax = fig.add_subplot(gs[1, :])
    sns.lineplot(iters, scores, ax=ax)
    plt.xlabel('iteration')
    plt.ylabel('score')
    ax.set_xlim((0, max_iter))
    ax.set_ylim((0, 1))


    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    img = Image.open(buffer)
    frames.append(img)


print('This will take a while...')
res = optimized_split(df, max_iter=max_iter,
                      n_partitions=4,
                      score_threshold=.999,
                      iter_callback=callback,
                      t_start=.5, t_decay=.995)

frames[0].save('balancing.gif', format='GIF',
                append_images=frames[1:], save_all=True, duration=350, loop=0)
