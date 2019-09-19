import torch
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
from unet3d import Unet3d
from unet_nopad import Unet3d_nopad
from preprocess import preprocessForNet1, preprocessForNet2
from predict import predict1, predict2


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL models
        # load net 1
        model1 = Unet3d(in_dim=5, out_dim=2, num_filter=16)
        net1 = model1.cpu()
        net1.load_state_dict(torch.load('model/net_step1.pth', map_location='cpu'))
        self._model1 = net1
        # load net 2
        model2 = Unet3d_nopad(in_dim=5, out_dim=4, num_filter=32)
        net2 = model2.cpu()
        net2.load_state_dict(torch.load('model/net_step2.pth', map_location='cpu'))
        self._model2 = net2


    def infer(self, input):
        # load preprocessed input
        inputAsNpArr1 = self._imageProcessor.loadAndPreprocess(input["t1"]["fileurl"], id="t1")
        inputAsNpArr2 = self._imageProcessor.loadAndPreprocess(input["t1c"]["fileurl"], id="t1c")
        inputAsNpArr3 = self._imageProcessor.loadAndPreprocess(input["t2"]["fileurl"], id="t2")
        inputAsNpArr4 = self._imageProcessor.loadAndPreprocess(input["flair"]["fileurl"], id="flair")
        # postprocessing
        print('loading done')
        affine = self._imageProcessor.returnAffine(input["t1"]["fileurl"])
        print('affine recovered')
        dataset = preprocessForNet1(inputAsNpArr1, inputAsNpArr2, inputAsNpArr3, inputAsNpArr4, affine)
        print('dataset assembled')
        print('first U-Net running')
        outNet1 = predict1(dataset, self._model1)
        print('preprocessing step 2')
        dataset = preprocessForNet2(dataset, outNet1[0])
        print('second U-Net running')
        outNet2 = predict2(dataset, self._model2)
        #outNet2 = predict2(outNet1, self._model2)
        print('done, postprocessing now')
        output = self._imageProcessor.computeOutput(outNet2[0])
        return output

    def _preprocessInputForNet(t1, t1c, t2, flair):
        pass
