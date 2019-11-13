import torch
import numpy as np
from torch.autograd import Variable
from misc import to_original_shape
import itertools
from skimage.measure import label, regionprops

def predict1(predict_sets, net):
    net.eval()
    for val_set in predict_sets:
        data = val_set['data']
        datashape = data.shape[1:]
        print(val_set['original_shape'])
        print(val_set['bbox_brain'])
        print(val_set['bbox_tumor'])
        print(val_set['affine'])

        zeropad_shape = np.ceil(np.divide(datashape, 8)).astype(np.int) * 8

        p = zeropad_shape - datashape  # padding
        p_b = np.ceil(p / 2).astype(np.int)  # padding before image
        p_a = np.floor(p / 2).astype(np.int)  # padding after image

        data_pad = np.pad(data, ((0, 0), (p_b[0], p_a[0]), (p_b[1], p_a[1]), (p_b[2], p_a[2])),
                          mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        inputs = data_pad[:5, :, :, :]  # just use t1, t2, flair
        print(inputs.shape)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs)
        print(inputs.shape)
        print('prediction1 is running')
        with torch.no_grad():
            outputs = net(inputs)

        print('reshaping predictions')
        # Bring predictions into correct shape and remove the zero-padded voxels
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        p_up = predictions.shape - p_a
        predictions = predictions[p_b[0]:p_up[0], p_b[1]:p_up[1], p_b[2]:p_up[2]]

        # Set all Voxels that are outside of the brainmask to 0
        mask = (data[0, :, :, :] != 0) | (data[1, :, :, :] != 0) | (data[3, :, :, :] != 0) | (data[4, :, :, :] != 0)
        predictions = np.multiply(predictions, mask)

        pred_orig_shape = to_original_shape(predictions, val_set)

        #save_nifti(opj(save_location, val_set['name'] + '_pred.nii.gz'), pred_orig_shape, val_set['affine'])

    return [pred_orig_shape, val_set['affine']]

def predict2(predict_sets, net):
    net = net.cpu()
    net.eval()

    for dataset in predict_sets:

        print("Processing " + dataset['name'] + "...")
        stride = 9
        if dataset['bbox_tumor'] == [0,0,0,0,0,0]:
            print("No tumor bbox found, setting whole brain as tumor mask!!!")

            dataset['data'][5] = (dataset['data'][0] != 0)

            brain_voxels = np.where(dataset['data'][0] != 0)
            minZidx = int(np.min(brain_voxels[0]))
            maxZidx = int(np.max(brain_voxels[0]))
            minXidx = int(np.min(brain_voxels[1]))
            maxXidx = int(np.max(brain_voxels[1]))
            minYidx = int(np.min(brain_voxels[2]))
            maxYidx = int(np.max(brain_voxels[2]))
            dataset['bbox_tumor'] = [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]

            stride = 9

        data = dataset['data']

        output_shape = [36,36,36]
        pad_net_shape = [44,44,44]

        pad = 80 # 36 + 44
        data_pad = np.pad(data, ((0, 0), (pad, pad), (pad, pad), (pad, pad)),
                          mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        complete_output = np.zeros(np.insert(data_pad.shape[1:],0,4), dtype=data.dtype)

        z1, z2, x1, x2, y1, y2 = dataset['bbox_tumor']

        start_points = itertools.product(range(z1,z2,stride), range(x1,x2,stride), range(y1,y2,stride))

        for z,x,y in start_points:
            z_start = z - pad_net_shape[0] + pad
            z_end = z + pad_net_shape[0] + output_shape[0] + pad
            x_start = x - pad_net_shape[1] + pad
            x_end = x + pad_net_shape[1] + output_shape[1] + pad
            y_start = y - pad_net_shape[2] + pad
            y_end = y + pad_net_shape[2] + output_shape[2] + pad

            input = data_pad[:5,z_start:z_end,x_start:x_end,y_start:y_end]
            input = np.expand_dims(input, axis=0)
            input_t = torch.from_numpy(input)
            input_t = Variable(input_t)

            with torch.no_grad():
                output_t = net(input_t)

            output = output_t.squeeze_(0).data.cpu().numpy()

            complete_output[:,z+pad:z+output_shape[0]+pad,x+pad:x+output_shape[1]+pad,y+pad:y+output_shape[2]+pad] = \
                complete_output[:,z + pad:z + output_shape[0] + pad, x + pad:x + output_shape[1] + pad, y + pad:y + output_shape[2] + pad] + output

        prediction = np.argmax(complete_output,axis=0)
        prediction = prediction[pad:-pad,pad:-pad,pad:-pad]
        prediction[prediction==3]=4

        # Keep only the values where the first step found a tumor
        prediction = np.multiply(prediction, data[5,:,:,:])

        pred_orig_shape = to_original_shape(prediction, dataset)

    return [pred_orig_shape, dataset['affine']]
