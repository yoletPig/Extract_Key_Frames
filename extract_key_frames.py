import numpy as np
from pathlib import Path
import math
import cv2
import numpy as np
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from auxfun_videos import VideoWriter
from skimage import io

videos = [
    # 'opencv\\20240313105042_NM_1203552_0708319856_1920_1080_25_06291456_0000000000041226_01_0305260036_04_312_0000000_V02_001_E00_0.avi'
    ]

def UniformFramescv2(cap, numframes2pick, start, stop, Index=None):
    """Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    nframes = len(cap)
    print(
        "Uniformly extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )

    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(nframes * stop), size=numframes2pick, replace=False
            )
        else:
            frames2pick = np.random.choice(
                range(math.floor(nframes * start), math.ceil(nframes * stop)),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))
        Index = np.array(Index, dtype=int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)
        
        
def KmeansbasedFrameselectioncv2(
    cap,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.

    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetitive.
    """
    nframes = len(cap)
    nx, ny = cap.dimensions
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    allocated = False
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(Index)
        
def extract_frames(
    algo="kmeans",
    numframes2pick = 5,
    start=0,
    stop=1,
    cluster_step=1,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=False,
    videos_list=None):
    
    has_failed = []
    for video in videos_list:
        cap = VideoWriter(video)
        nframes = len(cap)
        if not nframes:
            print("Video could not be opened. Skipping...")
            continue
        
        indexlength = int(np.ceil(np.log10(nframes)))
        
        fname = Path(video)
        output_path = Path("opencv") / fname.stem
        if not output_path.exists():
            output_path.mkdir()
        
        print("Extracting frames based on %s ..." % algo)
        if algo == "uniform":
            frames2pick = UniformFramescv2(cap, numframes2pick, start, stop)
        elif algo == "kmeans":
            frames2pick = KmeansbasedFrameselectioncv2(cap,numframes2pick,start,stop,step=cluster_step,resizewidth=cluster_resizewidth,color=cluster_color)
        else:
            print("Invalid algorithm. Skipping...")
            continue
        
        if not len(frames2pick):
            print("Frame selection failed...")
            return
                
        is_valid = []
        for index in frames2pick:
            cap.set_to_frame(index)  # extract a particular frame
            frame = cap.read_frame(crop=True)
            if frame is not None:
                image = img_as_ubyte(frame)
                img_name = (
                    str(output_path)
                    + "/img"
                    + str(index).zfill(indexlength)
                    + ".png"
                )
                io.imsave(img_name, image)
                is_valid.append(True)
            else:
                print("Frame", index, " not found!")
                is_valid.append(False)
        cap.close()
        
        if not any(is_valid):
            has_failed.append(True)
        else:
            has_failed.append(False)
    
    if all(has_failed):
        print("Frame extraction failed. Video files must be corrupted.")
        return
    elif any(has_failed):
        print("Although most frames were extracted, some were invalid.")
    else:
        print(
            "Frames were successfully extracted, for the videos listed in the config.yaml file."
        )
    print(
        "\nYou can now label the frames using the function 'label_frames' "
        "(Note, you should label frames extracted from diverse videos (and many videos; we do not recommend training on single videos!))."
    )    
    
    
if __name__ == "__main__":
    extract_frames(algo='kmeans',numframes2pick=500, videos_list=videos)