import numpy as np
import os
from scipy.optimize import leastsq
import pandas as pd
import rawpy
import re
import matplotlib.pyplot as plt
import cv2


def black_mean(black_path):
    black_imgs = [
        rawpy.imread(os.path.join(black_path, f)).raw_image.astype(np.float64)
        for f in os.listdir(black_path)
        if f.endswith(".dng")
    ]
    return np.mean(black_imgs, 0)


def save_fit_curve(x, y, param, label, path):
    plt.close()
    plt.scatter(x, y, s=1, c="red", alpha=0.5)
    plt.plot(param[0], param[1], alpha=0.5)
    # plt.plot(x, x, label='y=x', alpha=0.5)
    plt.title(label)
    plt.savefig(path)


def process_value(v_m, v_v, ch, v_mean, v_var, temp, mask=None, coords=None):
    if (v_m < 330 and (ch == 0 or ch == 3)) or (v_m < 210 and (ch == 1 or ch == 2)):
        v_mean.append(v_m)
        v_var.append(v_v)
    if (v_m >= 330 and (ch == 0 or ch == 3)) or (v_m >= 210 and (ch == 1 or ch == 2)):
        if mask is not None:
            temp[mask] = 1023
        elif coords is not None:
            temp[coords] = 1023


def process_imgs(imgs, subdir, ch, case_type):
    imgs_mean = np.mean(imgs, axis=-1).round().astype(np.uint16)
    temp = imgs_mean.copy()
    v_mean, v_var = [], []

    if case_type == 0:
        for h in range(imgs.shape[0]):
            for w in range(imgs.shape[1]):
                v_m = np.mean(imgs[h, w, :].flatten())
                v_v = np.var(imgs[h, w, :].flatten())
                process_value(v_m, v_v, ch, v_mean, v_var, temp, coords=(h, w))

    elif case_type == 1:
        for v_m in range(1024):
            mask = imgs_mean == v_m
            if mask.sum() > 10:
                v_v = np.var(imgs[mask, :].flatten())
                process_value(v_m, v_v, ch, v_mean, v_var, temp, mask=mask)

    os.makedirs(f"case_{case_type}", exist_ok=True)
    cv2.imwrite(
        os.path.join(f"case_{case_type}", subdir + str(ch // 2) + str(ch % 2) + ".png"),
        temp * 2**6,
    )

    return v_mean, v_var


def mean_var(src, black, name, case_type):
    ans = [[], [], [], []]

    for subdir in os.listdir(src):
        if re.match(r"^\d+$", subdir):
            dngs_path = os.path.join(src, subdir, "0000")
            for ch in range(4):
                imgs = [
                    rawpy.imread(os.path.join(dngs_path, f)).raw_image.astype(
                        np.float64
                    )
                    - black
                    for f in os.listdir(dngs_path)
                    if f.endswith(".dng")
                ]
                imgs = [
                    np.expand_dims(img[20:-20, 20:-20][ch // 2 :: 2, ch % 2 :: 2], -1)
                    for img in imgs
                ]
                imgs = np.clip(np.concatenate(imgs, -1), 0, 1023)
                v_mean, v_var = process_imgs(imgs, subdir, ch, case_type)
                k, b = np.polyfit(v_mean, v_var, 1)
                ans[ch].append([float(subdir) / 16, round(k, 3), round(b, 3)])
                save_fit_curve(
                    v_mean,
                    v_var,
                    [[0, 1023], [b, 1023 * k + b]],
                    str(round(k, 3)) + " * T + " + str(round(b, 3)),
                    os.path.join("./", name, f"{ch//2}{ch%2}", subdir + ".png"),
                )
    ans = [pd.DataFrame(columns=["gain", "a", "b"], data=ans[i]) for i in range(4)]
    return ans


def plot_fit(ans, name):
    for ch in range(4):
        ex_list = ans[ch].astype("float")
        if len(ex_list["gain"]) > 3:
            x = ex_list["gain"].to_list()
            pX = np.linspace(0, ex_list["gain"].max())
            # poisson
            y = ex_list["a"].to_list()
            f = np.poly1d(np.polyfit(x, y, 1))
            label = str(f)
            print(label)
            save_fit_curve(
                x,
                y,
                [pX, f(pX)],
                label,
                os.path.join("./", name, f"{ch//2}{ch%2}", f"a.png"),
            )
            # gauss
            y = ex_list["b"].to_list()
            func2 = lambda a, x: a[0] * x * x + a[1]
            error2 = lambda a, x, y: func2(a, x) - y
            param = leastsq(error2, [1, 1], args=(np.array(x), np.array(y)))
            label = (
                str(round(param[0][0], 6)) + " * T^2 + " + str(round(param[0][1], 6))
            )
            print(label)
            save_fit_curve(
                x,
                y,
                [pX, param[0][0] * pX * pX + param[0][1]],
                label,
                os.path.join("./", name, f"{ch//2}{ch%2}", f"b.png"),
            )


if __name__ == "__main__":
    src = "/share/noise_calib_0709/"
    case_type = 1
    name = src.split("/")[-2] + "_" + str(case_type)
    for ch in range(4):
        os.makedirs(os.path.join("./", name, f"{ch//2}{ch%2}"), exist_ok=True)
    ans = mean_var(src, 64.0, name, case_type)
    plot_fit(ans, name)
