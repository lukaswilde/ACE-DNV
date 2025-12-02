# This file is responsible for reading specified recordings from the Gaze-in-Wild dataset and doing preprocessing on them.
# the output of this function is the dataframe ready for next machine learning steps.
# in the main directory of the datasetyou need a txt file of list of recordings that you specified.
###The name of the txt file list is given in the config file as RECS_LIST variable. It has a structure as follwing:
#### participantNum, TaskNum, TaskName
#### for instance:
# 1,1,Indoor_Walk
# 2,1,Indoor_Walk
# 3,1,Indoor_Walk


from os import path

import numpy as np
from config import DATA_DIR, EXP, INP_DIR, LABELER, OUT_DIR, RECS_LIST
from modules.PatchSimNet import pred_all
from modules.preprocess import preprocessor
from scipy.io import loadmat


def GiWPrep():
    ds_x = []
    ds_y = []

    # list the files inside the input directory
    rec_list = np.loadtxt(INP_DIR + RECS_LIST, delimiter=",", dtype="str")
    print(f"Found {len(rec_list)} relevant recordings")

    for p in range(len(rec_list)):
        activityName = rec_list[p, 2]
        activityNum = rec_list[p, 1]
        participantNum = rec_list[p, 0]

        # video path
        vidPath = DATA_DIR + f"videos_kothari/PrIdx_{participantNum}_TrIdx_{activityNum}_world.mp4"
        processData = loadmat(
            DATA_DIR + f"signals_kothari/PrIdx_{participantNum}_TrIdx_{activityNum}.mat",
        )
        labels = loadmat(
            DATA_DIR
            + f"labels_kothari/PrIdx_{participantNum}_TrIdx_{activityNum}_Lbr_{LABELER}.mat",
        )
        labels = np.array(labels["LabelData"]["Labels"][0])
        frames = processData["ProcessData"]["ETG"][0, 0][0, 0][5][0]

        # finding the indices that match with frames
        frames, indcs = np.unique(frames, return_index=True)

        # time points
        T = np.squeeze(np.array(processData["ProcessData"]["T"][0, 0]))

        # take the gazes out
        gazes = np.array(
            processData["ProcessData"]["ETG"][0, 0][0, 0][8]
            * processData["ProcessData"]["ETG"][0, 0][0, 0][0],
        )

        visod = np.loadtxt(INP_DIR + f"{participantNum}/{activityNum}/visOdom.txt", delimiter=" ")

        headRot = visod[:, 5]
        bodyMotion = visod[:, 1:3]

        frameRange = np.loadtxt(
            INP_DIR + f"{participantNum}/{activityNum}/range.txt",
            delimiter=" ",
        )
        startFrame = frameRange[0] + 1
        endFrame = frameRange[1]

        # if not all headRot were computed we trim until where available
        rm_indcs = np.where(frames >= len(headRot) + startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])

        # if we started from a specific frame remove the previous ones
        rm_indcs = np.where(frames < startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])

        # match the gazes that fall into frames
        gazeMatch = gazes[indcs]

        # reload gazed in [0, 1]
        gazes = np.array(processData["ProcessData"]["ETG"][0, 0][0, 0][8])

        # keep the labels that correspond to a frame
        labels = labels[0]
        labels = np.squeeze(labels)

        lblMatch = labels[indcs]
        TMatch = T[indcs]

        headRot = headRot[frames[:-1] - (int(startFrame) - 1)]
        bodyMotion = bodyMotion[frames[:-1] - (int(startFrame) - 1)]

        # through our timestamps in the gaze in the beginning
        gazes = gazes[TMatch[0] < T]
        labels = labels[TMatch[0] < T]
        T = T[TMatch[0] < T]

        # through our timestamps in the gaze in the end
        gazes = gazes[TMatch[-1] > T]
        labels = labels[TMatch[-1] > T]
        T = T[TMatch[-1] > T]

        if not path.exists(INP_DIR + participantNum + "/" + str(activityNum) + "/patchDists.csv"):
            patchDists = pred_all(vidPath, gazeMatch, frames)
            np.savetxt(
                INP_DIR + participantNum + "/" + str(activityNum) + "/patchDists.csv",
                patchDists,
                delimiter=",",
            )
        else:
            patchDists = np.loadtxt(
                INP_DIR + participantNum + "/" + str(activityNum) + "/patchDists.csv",
                delimiter=",",
            )

        patchDists = np.transpose(np.array(patchDists))

        feats, lbls = preprocessor(
            gazes,
            patchDists,
            headRot,
            bodyMotion,
            T,
            TMatch,
            labels,
            lblMatch,
        )

        print(
            "number of fixations "
            + str(len(np.where(lbls == 0)[0]))
            + ", saccades "
            + str(len(np.where(lbls == 2)[0]))
            + ", gazeP "
            + str(len(np.where(lbls == 1)[0]))
            + ", gazeF "
            + str(len(np.where(lbls == 3)[0])),
        )

        if EXP == "2-2":
            lbls[np.where(lbls == 3)] = 0

        # some tasks do not contain some labels. But you might see that few of them exist with
        # mislabeling. So here, we clean them so that it would not cause fault in ML classifer
        if EXP == "1-1" or EXP == "1-3":
            # temp delete fixation
            rmInd = np.where(lbls == 0)[0]
            lbls = np.delete(lbls, rmInd)
            feats = np.delete(feats, rmInd, axis=0)

        if EXP == "1-1" or EXP == "1-3":
            # temp delete gaze p
            rmInd = np.where(lbls == 1)[0]
            lbls = np.delete(lbls, rmInd)
            feats = np.delete(feats, rmInd, axis=0)
        if EXP == "1-2":
            # temp delete GFo
            rmInd = np.where(lbls == 3)[0]
            lbls = np.delete(lbls, rmInd)
            feats = np.delete(feats, rmInd, axis=0)

        if EXP == "1-1" or EXP == "1-3":
            lbls[lbls == 2] = 0
            lbls[lbls == 3] = 1

        ds_x.append(feats)
        ds_y.append(lbls)

        np.savetxt(
            OUT_DIR
            + "feats_p"
            + str(participantNum)
            + "_a"
            + str(activityNum)
            + "_l"
            + str(LABELER)
            + ".csv",
            feats,
            delimiter=",",
        )
        np.savetxt(
            OUT_DIR
            + "lbls_p"
            + str(participantNum)
            + "_a"
            + str(activityNum)
            + "_l"
            + str(LABELER)
            + ".csv",
            lbls,
            delimiter=",",
        )
        np.savetxt(
            OUT_DIR + "frames_p" + str(participantNum) + "_a" + str(activityNum) + ".csv",
            frames,
            delimiter=",",
        )
        np.savetxt(
            OUT_DIR + "gazes_p" + str(participantNum) + "_a" + str(activityNum) + ".csv",
            gazeMatch,
            delimiter=",",
        )
        print(
            "Data successfully loaded for participant "
            + str(participantNum)
            + " task: "
            + str(activityNum),
        )

    return ds_x, ds_y
