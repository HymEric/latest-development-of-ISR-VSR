Mainly ICCV, ECCV and CVPR about ISR and VSR, especially lasted two years developments.

There are some useful repositories thst help me a lot：  
1、[A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.](https://github.com/LoSealL/VideoSuperResolution) which has most great papers/models about ISR and VSR. Includ some useful tools: aome model pre-trained weights, Link of datasets, VSR package which offers a training and data processing framework based on TF.  
2、[Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution) which has some VSR paper's informations.

Thanks them. But above repositories are not completely, and most of ECCV2018 and CVPR2018 are not listed as well as my under-graduation project is doing, therefore, this repositories exists. (help me and you) I holp it will be updating under our contributions.

## ISR: ##
<table>
    <tr>
        <th>abbreviation</th>
		<th>full name</th>
		<th>published</th>
		<th>code</th>
		<th>description</th>
		<th>keywords</th>
		<th>my project*</th>
    </tr>
    <tr>
        <td>SRCNN</td>
        <td>Image Super-Resolution Using Deep Convlutional Network</td>
        <td>ECCV2014</td>
		<td>keras：https://github.com/qobilidop/srcnn</td>
        <td>has two version 2014 and ex-2016. Milestone in deep learning about SR.Simple three CNN network：patch extraction and representation, non-linear mapping and reconstraction</td>
		<td>Loss:MSE CNN</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FSRCNN</td>
        <td>Accelerating the Super-Resolution Convolutional Neural Network</td>
        <td>ECCV2016</td>
		<td>official:matlab,caffe：http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.htmlhttp://</td>
        <td>Develop SRCNN, add deconv, input image don't need to upsample by bicubic and fine-tune accelerate</td>
		<td>deconvolution fine-tuninig last deconv</td>
        <td>*</td>
    </tr>
    <tr>
        <td>ESPCN
</td>
        <td>Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
</td>
        <td>CVPR2016
</td>
		<td>github(tensorflow): https://github.com/drakelevy/ESPCN-TensorFlowhttps://
github(pytorch): https://github.com/leftthomas/ESPCNhttps://
github(caffe): https://github.com/wangxuewen99/Super-Resolution/tree/master/ESPCNhttps://</td>
        <td>A new way to upsamping: sub-pixel</td>
		<td>sub-pixel Tanh instead Relu Real time</td>
        <td>*</td>
    </tr>
</table>

## VSR ##

## About Merics ##
"The Perception-Distortion Tradeoff"
