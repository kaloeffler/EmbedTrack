"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
"""
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class AverageMeter(object):
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)
            ]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Logger:
    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print("Created logger with keys:  {}".format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        count = 0
        for key in self.data:
            if count < 3:
                keys.append(key)
                data = self.data[key]
                ax.plot(range(len(data)), data, marker=".")
                count += 1
        ax.legend(keys, loc="upper right")
        ax.set_title(self.title)

        plt.draw()
        plt.close(fig)
        self.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + ".png"))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + ".csv"))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams["backend"]
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return
