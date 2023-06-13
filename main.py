from concurrent import futures
import cv2
import glob
import sys


def matching(src, tmp, mask):
    result = cv2.matchTemplate(src, tmp, cv2.TM_CCORR_NORMED, mask=mask)
    # _best, _, _bestPos, _ = cv2.minMaxLoc(result)  # min
    _, _best, _, _bestPos = cv2.minMaxLoc(result)  # max
    return _best, _bestPos


level = sys.argv[1]
srcs_path = glob.glob(level + "/final/*.ppm")
tmps_path = glob.glob(level + "/*.ppm")

srcs_name = [src.split("/")[2][0:-4] for src in srcs_path]
tmps_name = [tmp.split("/")[1][0:-4] for tmp in tmps_path]

src = [cv2.imread(src,0) for src in srcs_path]
tmp = [[cv2.imread(tmp,0)] for tmp in tmps_path]

num = 6

with futures.ThreadPoolExecutor(max_workers=64) as executor:
    for i in range(len(tmp)):
        tmp[i].append(cv2.resize(tmp[i][0], None, None, 0.5, 0.5))
        tmp[i].append(cv2.resize(tmp[i][0], None, None, 2.0, 2.0))
        tmp[i].append(cv2.rotate(tmp[i][0], cv2.ROTATE_90_CLOCKWISE))
        tmp[i].append(cv2.rotate(tmp[i][0], cv2.ROTATE_180))
        tmp[i].append(cv2.rotate(tmp[i][0], cv2.ROTATE_90_COUNTERCLOCKWISE))
    mask = [
        [cv2.threshold(_tmp[i], 0, 255, cv2.THRESH_BINARY)[1] for i in range(num)]
        for _tmp in tmp
    ]

    result_list = []

    for i in range(len(src)):
        for j in range(len(tmp)):
            for k in range(num):
                result = executor.submit(matching, src[i], tmp[j][k], mask[j][k])
                result_list.append(result)

    for i in range(len(src)):
        best = -float("inf")
        bestPos = [0, 0]
        idx = 0
        kdx = 0
        for j in range(len(tmp)):
            for k in range(num):
                _best, _bestPos = result_list[i * len(tmp) * num + j * num + k].result()
                if best < _best:
                    best = _best
                    bestPos = _bestPos
                    idx = j
                    kdx = k
        file = open("result/" + srcs_name[i] + ".txt", "w")
        file.write(
            "{} {} {} {} {} {}".format(
                tmps_name[idx],
                bestPos[0],
                bestPos[1],
                tmp[idx][kdx].shape[1],
                tmp[idx][kdx].shape[0],
                0 if (kdx < 3) else (kdx - 2) * 90,
            )
        )
