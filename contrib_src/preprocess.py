import numpy as np
from skimage.measure import label, regionprops

def normalize(image, mask=None):
    if mask is None:
        mask = image != image[0,0,0]

    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - image[mask].mean()) / image[mask].std()
    return image

def preprocessForNet1(t1, t1ce, t2, flair, affine):
    assert t1[0, 0, 0] == 0, 'non-zero background?!'
    assert t2[0, 0, 0] == 0, 'non-zero background?!'
    assert t1ce[0, 0, 0] == 0, 'non-zero background?!'
    assert flair[0, 0, 0] == 0, 'non-zero background?!'

    t1ce_sub_t1 = t1ce - t1

    #%% brain mask
    mask = (t1 != 0) | (t1ce != 0) | (t2 != 0) | (flair != 0)

    #%% Extract Brain BBox
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    t1 = normalize(t1)
    t2 = normalize(t2)
    t1ce = normalize(t1ce)
    t1sub = normalize(t1ce_sub_t1)
    flair = normalize(flair)

    t1_boxed = t1[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
    t2_boxed = t2[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
    t1ce_boxed = t1ce[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
    t1sub_boxed = t1sub[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]
    flair_boxed = flair[minZidx:maxZidx,minXidx:maxXidx,minYidx:maxYidx]

    all_data = np.zeros([6] + list(t1_boxed.shape), dtype=np.float32)
    all_data[0] = t1_boxed
    all_data[1] = t1ce_boxed
    all_data[2] = t1sub_boxed
    all_data[3] = t2_boxed
    all_data[4] = flair_boxed
    print('Sanity check on boxed data:')
    for elem in all_data:
        print('Element has shape {}'.format(elem.shape))
    #np.save(opj(savedir, subject + ".npy"), all_data)

    bbox_brain = [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]
    bbox_tumor = [0,0,0,0,0,0]
    info = {'original_shape': t1.shape, 'bbox_brain': bbox_brain, 'bbox_tumor': bbox_tumor, 'name': 'Segmentation', 'affine': affine}
    #pickle.dump(info, open(opj(savedir, subject + ".pkl"), "wb"))

    dataset = []
    info['data'] = all_data
    dataset.append(info)

    return dataset

def preprocessForNet2(data, resultNet1):
    #preprocs = load_brats.load_normalized_data(dir=savedir_preproc)
    preprocs = data

    # Load results from step1
    prediction = resultNet1

    for idx, dataset in enumerate(preprocs):
        print('in loop now')
        name = dataset['name']

        #%% Keep only the biggest found region in the brain
        cc = dataset['bbox_brain']
        print('reshaping prediction')
        pred_in_shape = prediction[cc[0]:cc[1],cc[2]:cc[3],cc[4]:cc[5]]
        biggest_size = -1
        biggest_region = 0
        print('iterating through regions')
        for region in regionprops(label(pred_in_shape)):
            if region.area > biggest_size:
                biggest_size = region.area
                biggest_region = region

        if biggest_region == 0:
            print("ATTENTION, No Tumor found in image: " + name)
            continue

        z1, x1, y1, z2, x2, y2 = biggest_region.bbox
        bbox_tumor = [z1,z2,x1,x2,y1,y2]
        dataset['bbox_tumor'] = bbox_tumor

        dataset['data'][5,:,:,:] = np.zeros(dataset['data'].shape[1:]) # if train set, delete original segmentation
        for p in biggest_region.coords:
            dataset['data'][5,p[0],p[1],p[2]] = 1

    return preprocs
