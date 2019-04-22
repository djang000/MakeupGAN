# MakeupGAN
This is an implementation of a unpaired MakeUp Generator Architecture for GAN on tensorflow. The model generates makeup image from no-makeup face or vice versa.

# Getting Started
* ([BeautyGAN.py](/libs/network/BeautyGAN.py), [config.py](/libs/configs/config.py)): These files are the main MakeupGAN network.

* ([datapipe.py](/datasets/datapipe.py)): This file's role is loading and changing to tensor your dataset taht are in makeup_dataset folder which is consist of image and segs(mask) folder. The image folder contain makeup image and no-makeup image folder and the segs folder have 3 partial region that are eye, shadow, lips.

* ([pretrained_models](/pretrained_models)): you put the pretrained ([VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)) model in this folder.

* ([train.py](/train.py)): this file is for training.
	To train this network
''' python train.py '''

* ([inference.py](/inference.py)): this file is for inference. we have 2 inference mode that are reference style and random style and you set True or False on --rand_style value.
	To run this file
''' python infernece.py --rand_style=True'''

# Result
<table >
    <tr >
    	<td><center>no-makeup</center></td>
        <td><center>makeup</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="/result/case2/oriA.jpg"></center>
    	</td>
    	<td>
    		<center><img src="/result/case2/oriB.jpg"></center>
    	</td>
    </tr>
    <tr >
        <td><center>Generated makeup image</center></td>
        <td><center>Generated no-makeup image</center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="/result/case2/fake_AB.jpg"></center>
        </td>
        <td>
        	<center><img src="/result/case2/fake_BA.jpg"></center>
        </td>
    </tr>
</table>


