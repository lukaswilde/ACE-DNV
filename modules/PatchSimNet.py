import os

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchfile
from config import CH2STREAM_MODEL_DIR, LAMBDA, PATCH_PRIOR_STEPS, PATCH_SIZE
from modules.patchExtractor import patchExtractor
from torch.autograd import Variable

CUDA_ID = "0"
model = "2ch2stream"
lua_model = CH2STREAM_MODEL_DIR


def conv2d(input, params, base, stride=1, padding=0):
    return F.conv2d(input, params[base + ".weight"], params[base + ".bias"], stride, padding)


def linear(input, params, base):
    return F.linear(input, params[base + ".weight"], params[base + ".bias"])


#####################   2ch2stream   #####################


def deepcompare_2ch2stream(input, params):
    def stream(input, name):
        o = conv2d(input, params, name + ".conv0")
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + ".conv1")
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + ".conv2")
        o = F.relu(o)
        o = conv2d(o, params, name + ".conv3")
        o = F.relu(o)
        # return o.view(o.size(0), -1)
        return torch.flatten(o)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), "fovea")
    o_retina = stream(F.pad(input, (-16,) * 4), "retina")
    o = linear(torch.cat([o_fovea, o_retina], dim=0), params, "fc0")
    return linear(F.relu(o), params, "fc1")


model = deepcompare_2ch2stream

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.mps.is_available() else device


def create_network():
    print(f"USING DEVICE {device}")

    def cast(t):
        return torch.from_numpy(t).float().to(device)  # if torch.cuda.is_available() else t

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
    if torch.cuda.is_available():
        # to prevent opencv from initializing CUDA in workers
        torch.randn(8).cuda()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    net = torchfile.load(lua_model)
    params = {}
    for j, branch in enumerate(["fovea", "retina"]):
        counter = 0
        for k, layer in enumerate(net.modules[0].modules[j].modules[1].modules):
            if layer.weight is not None:
                if k in [1, 4, 7, 9]:
                    params["%s.conv%d.weight" % (branch, counter)] = layer.weight
                    params["%s.conv%d.bias" % (branch, counter)] = layer.bias
                    counter = counter + 1

    counter = 0
    for k, layer in enumerate(net.modules):
        if layer.weight is not None:
            if k in [1, 3]:
                params["fc%d.weight" % counter] = layer.weight
                params["fc%d.bias" % counter] = layer.bias
                counter = counter + 1

    params = {k: Variable(cast(v)) for k, v in params.items()}

    return params


def perdict(input, params):
    y = model(torch.from_numpy(input).float().to(device), params)

    return y


def pred_all(vidDir, gazes, target_frames):
    t = 1
    cap = cv.VideoCapture(vidDir)
    cap.set(cv.CAP_PROP_POS_FRAMES, target_frames[1] - 2)
    ret, frame = cap.read()
    f = target_frames[1] - 2

    prvFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # color conversion to grayscale
    prvPatch = patchExtractor(prvFrame, gazes[0])
    patchSimNet_params = create_network()
    dists = []
    normDists = []
    while True:
        ret, frame2 = cap.read()

        if (not ret) or (t == (len(target_frames))):
            print("Patch similarities computed successfully!")
            break

        nxtFrame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        nxtPatch = patchExtractor(nxtFrame, gazes[t])

        if f % 250 == 0:
            print(f)

        inp = np.zeros((2, PATCH_SIZE, PATCH_SIZE))
        if (
            (prvPatch.shape[0] == 0)
            or (prvPatch.shape[1] == 0)
            or (nxtPatch.shape[0] == 0)
            or (nxtPatch.shape[1] == 0)
        ):
            dists.append(0)
            normDists.append(0)  # TODO handle fault
            f = f + 1
            t += 1
            prvPatch = nxtPatch
            continue

        inp[0, :, :] = cv.resize(prvPatch, (64, 64))
        inp[1, :, :] = cv.resize(nxtPatch, (64, 64))
        patchDist = perdict(inp, patchSimNet_params)
        dists.append(patchDist.item())
        pastPatchDist = np.sum(dists[-PATCH_PRIOR_STEPS:])
        patchDistAvg = (1 - LAMBDA) * patchDist.item() + LAMBDA * (
            pastPatchDist / PATCH_PRIOR_STEPS
        )
        assert patchDistAvg == patchDist.item()
        normDists.append(patchDistAvg)

        f = f + 1
        t += 1
        prvPatch = nxtPatch

    print("Finished!")
    return normDists
