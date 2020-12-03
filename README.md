# Latest-development-of-ISR-VSR

**[Updating...]** Papers and **related resources**, mainly state-of-the-art and novel works in ICCV, ECCV and CVPR about image super-resolution and video super-resolution.

## Contents
<!-- TOC -->

- [1. Metrics dispute](#Metrics-dispute)
- [2. Latest survey](#Latest-survey) 
- [3. Upscale method](#Upscale-method)
- [4. Unsupervised Super-Resolution Method](#Unsupervised-Super-Resolution-Method)
- [5. Real-Word Image Super-Resolution](#Real-Word-Image-Super-Resolution)
- [6. Stereo Image Super-Resolution](#Stereo-Image-Super-Resolution)
- [7. **Image-Super-Resolution (ISR)**](#Image-Super-Resolution)
- [8. **Video-Super-Resolution (VSR)**](#Video-Super-Resolution)
- [9. Library](Library)
- [10. Related Research institutions](#Related-Research-institutions)
<!-- meta-data related -->
<!-- knowledge distillation related -->
<!-- TOC -->

## Metrics dispute
Suggestion in SR: CVPR2018 ["The Perception-Distortion Tradeoff"](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.06077)

## Latest survey
- [Deep Learning for Image Super-resolution: A Survey](https://arxiv.org/abs/1902.06068), arXiv [Submitted on 16 Feb 2019 (v1), last revised 8 Feb 2020 (this version, v2)], accepted by PAMI2020
- [A Deep Journey into Super-resolution: A survey](https://arxiv.org/abs/1904.07523), arXiv, [Submitted on 16 Apr 2019 (v1), last revised 23 Mar 2020 (this version, v3)]
- [Deep learning methods in real‑time image super‑resolution: a survey](https://link.springer.com/article/10.1007/s11554-019-00925-3), Journal of Real-Time Image Processing2020 
- [Survey on Single Image based Super-resolution—Implementation Challenges and Solutions](https://link.springer.com/article/10.1007/s11042-019-08254-0), Multimedia Tools and Applications2020

## Upscale method
- Dconvolution: ["Deconvolutional networks"](https://ftp.cs.nyu.edu/~fergus/papers/matt_cvpr10.pdf)
- Sub-pixel: ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/abs/1609.05158)
- Unpooling: ["Visualizing and understanding convolutional networks"](https://arxiv.org/abs/1311.2901)
- DUpsample: ["Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation"](https://arxiv.org/abs/1903.02120)
- Carafe: ["CARAFE- Content-Aware ReAssembly of FEatures"](https://arxiv.org/abs/1905.02188)
- Meta-SR: ["Meta-SR-A Magnification-Arbitrary Network for Super-Resolution"](https://arxiv.org/abs/1903.00875)
- Scale-arbitrary SR:[Learning for Scale-Arbitrary Super-Resolution from Scale-Specific Networks](https://arxiv.org/abs/2004.03791), arXiv2020

## Unsupervised Super-Resolution Method
1. ["Zero-Shot" Super-Resolution using Deep Internal Learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html), CVPR2018
2. [Unsupervised image super-resolution using cycle-in-cycle generative adversarial networks](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Yuan_Unsupervised_Image_Super-Resolution_CVPR_2018_paper.html), CVPRW2018
3. [Adversarial training with cycle consistency for unsupervised super-resolution in endomicroscopy](https://www.sciencedirect.com/science/article/pii/S1361841518305966), Medical image analysis 2019
4. [Self-Supervised Fine-tuning for Image Enhancement of Super-Resolution Deep Neural Networks](https://arxiv.org/abs/1912.12879), arXiv2019
5. [Unsupervised Learning for Real-World Super-Resolution](https://arxiv.org/abs/1909.09629), arXiv2019
6. [Unsupervised Single-Image Super-Resolution with Multi-Gram Loss](https://www.mdpi.com/2079-9292/8/8/833), MDPI2019

## Real-Word Image Super-Resolution
 - **Based on the proposed HR-LR Image Pairs**
1. [Toward Bridging the Simulated-to-Real Gap: Benchmarking Super-Reslution on Real Data](https://arxiv.org/abs/1809.06420v2), TPAMI2019
2. [Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](http://openaccess.thecvf.com/content_ICCV_2019/html/Cai_Toward_Real-World_Single_Image_Super-Resolution_A_New_Benchmark_and_a_ICCV_2019_paper.html),ICCV2019
3. [Camera Lens Super-Resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper), CVPR2019 
4. [Zoom to Learn, Learn to Zoom](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html), CVPR2019 
- **Based on the simulated degradation method**
1. [Blind Super-Resolution with Iterative Kernel Corrections](http://openaccess.thecvf.com/content_CVPR_2019/html/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.html), CVPR2019
2. [Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.html), CVPR2019
3. [Blind Super-Resolution Kernel Estimation using an Internal-GAN](http://papers.nips.cc/paper/8321-blind-super-resolution-kernel-estimation-using-an-internal-gan), NeurIPS2019
4. [Kernel Modeling Super-Resolution on Real Low-Resolution Images](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.html), ICCV2019

## Stereo Image Super-Resolution
1. [Enhancing the Spatial Resolution of Stereo Images using a Parallax Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Jeon_Enhancing_the_Spatial_CVPR_2018_paper.html), CVPR2018
<!-- StereoSR,one left LR and one right LR as inputs, but 64 copies of right LR before to luminance net, first learn luminance then to map to RGB by chrominance net, YCbCr to RGB -->
2. [Learning Parallax Attention for Stereo Image Super-Resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Parallax_Attention_for_Stereo_Image_Super-Resolution_CVPR_2019_paper.html), CVPR2019
<!-- PASSRnet, proposed PAM (parallax attention modual), new Flicker1024 datasets, extend to another: Parallax-based Spatial and Channel Attention Stereo SR network paper by it -->
3. [Stereoscopic Image Super‑Resolution with Stereo Consistent Feature](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-SongW.10348.pdf), AAAI2020 oral
<!-- SPAMnet, Self and Parallax Attention Mechanism (SPAM), new loss: Stereo-consistency Loss for stereo consistence, disparity map-->
4. [A Stereo Attention Module for Stereo Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/8998204/), SPL2020
<!-- SAM (Stereo attention module), SAM can inset to any SR model, fine-tune after inserting SAM -->

## Image Super-Resolution
Sorted by year and the format is: abbreviation, paper title, publicaiton, [highlights], related source code. 

##### In 2020
- IPT, [Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364), **arXiv**, [low-level transformer]  
[waiting]() 

- IGNN, [Cross-Scale Internal Graph Neural Network for Image Super-Resolution](http://proceedings.neurips.cc/paper/2020/hash/23ad3e314e2a2b43b4c720507cec0723-Abstract.html), **NeurIPS2020**, [graph related, patch match]  
[codes](https://github.com/sczhou/IGNN)

- SRFlow, [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://arxiv.org/abs/2006.14200), **ECCV2020**  
[codes-prepare](https://github.com/andreas128/SRFlow)

- PISR, [Learning with Privileged Information for Efficient Image Super-Resolution](https://arxiv.org/abs/2007.07524), **ECCV2020**, [use encoder and decoder in teacher, distillation, estimator module]  
[pytorch-codes](https://github.com/cvlab-yonsei/PISR)

- [Coarse-to-fine cnn for image super-resolution](https://ieeexplore.ieee.org/abstract/document/9105085/), **IEEE TMM2020**  
[pytorch-codes](https://github.com/hellloxiaotian/CFSRCNN)

- [Journey Towards Tiny Perceptual Super-Resolution](https://arxiv.org/abs/2007.04356), **ECCV2020**

- [Lightweight Image Super-Resolution with Enhanced CNN](https://arxiv.org/abs/2007.04344), **arXiv2020**, Elsevier  
[pytorch-codes](https://github.com/hellloxiaotian/LESRCNN)

- [Unpaired Image Super-Resolution using Pseudo-Supervision](http://openaccess.thecvf.com/content_CVPR_2020/html/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.html), **CVPR2020**

- [Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline](http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline_CVPR_2020_paper.html), **CVPR2020**  
[tensorflow-codes](https://github.com/alex04072000/SingleHDR)

- [Invertible Image Rescaling](https://arxiv.org/abs/2005.05650), **ECCV2020**, [another method to get more information in the sacaling phase, invertible NN, flow-based, wavelet transform]  
[codes](https://github.com/pkuxmq/Invertible-Image-Rescaling)

- IGNN, [Cross-Scale Internal Graph Neural Network for Image Super-Resolution](https://arxiv.org/abs/2006.16673), **arXiv2020**, [first use the graph neural network, graph construction and patch aggreagation module, find the k similar neighbor patch]  
[codes](https://github.com/sczhou/IGNN)

- TTSR, [Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139), **CVPR2020**, [proposed a transformer-based model to do SR, texture transformer]  
[codes-wait]

- CutBlur, [Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy](https://arxiv.org/abs/2004.00448), **CVPR2020**, [new data augmentation method called CutBlur, it not only can tackle SR but other low-level tack like denoising and artifact ramoval, cut-and-paste based on patch, let model to know where to SR and how to SR]  
[pytorch-codes](https://github.com/clovaai/cutblur)


- SPSR, [Structure-Preserving Super Resolution with Gradient Guidance](https://arxiv.org/abs/2003.13081), **CVPR2020**, [Gradient guidance to perserve the information, gradient loss, address the geometric distort]  
[pytorch-codes](https://github.com/Maclory/SPSR)


- UDVD, [Unified Dynamic Convolutional Network for Super-Resolution with Variational Degradations](https://arxiv.org/abs/2004.06965), **CVPR2020**, [try to use one model to address several degreadation, Feature Extraction Network(FRN), Refinement Network(RN), Dynamic Block(DB), dynamic conv by a dynamic kernels some like sub-pixel operation]   
\-\-\- 


##### In 2019
- SRFBN, [Feedback Network for Image Super-Resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.html), **CVPR2019**, [feedback and a lot of comparation]  
[pytorch-codes](https://github.com/Paper99/SRFBN_CVPR19)


- zoom-learn-zoom, [Zoom to Learn, Learn to Zoom](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html), **CVPR2019**, [SR-RAW dataset and CoBi loss, real-word, new direction for SR-RAW datasets and new CoBi loss function for alignment]  
[tensorflow-codes](https://github.com/ceciliavision/zoom-learn-zoom)


- Camera, [Camera Lens Super-Resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper), **CVPR2019**, [real-word, Create City100 Dataset for real-word application]  
[tensorflow-codes](https://github.com/ngchc/CameraSR)


- RealSR, [Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](http://openaccess.thecvf.com/content_ICCV_2019/html/Cai_Toward_Real-World_Single_Image_Super-Resolution_A_New_Benchmark_and_a_ICCV_2019_paper.html), **ICCV2019**, [RealSR dataset, real-word, LP-KPN, New RealSR datasets more flexible and convenient to use]  
[caffe-codes](https://github.com/csjcai/RealSR)


- Simulated-to-Real Gap, [Toward Bridging the Simulated-to-Real Gap: Benchmarking Super-Reslution on Real Data](https://ieeexplore.ieee.org/abstract/document/8716546/), **TPAMI2019**, [hardware binning, real-word, maybe the method older for it's journal]  
\-\-\-


- RankSRGAN, [RankSRGAN: Generative Adversarial Networks with Ranker for Image Super- Resolution](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_RankSRGAN_Generative_Adversarial_Networks_With_Ranker_for_Image_Super-Resolution_ICCV_2019_paper.html), **CVPR2019**, [focus on perceptual quality, and new method to use perceptual metrics named Ranker]  
[pytorch-codes](https://github.com/WenlongZhang0724/RankSRGAN)


- IMDN, [Lightweight Image Super-Resolution with Information Multi-distillation Network](https://dl.acm.org/doi/abs/10.1145/3343031.3351084), **ACM MM2019**  
[pytorch-codes](https://github.com/Zheng222/IMDN)

##### In 2018
- WDSR, [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718), **arXiv2018**, [widen feature map and WN, weight normalization]  
[pytorch-codes](https://github.com/JiahuiYu/wdsr_ntire2018)


- SRMD, [Learning a Single Convolutional Super-Resolution Network for Multiple Degradations](), **CVPR2018**, Degraded Fuzzy Kernel and Noise Level  
[matlab-codes](https://github.com/cszn/SRMD)


- RDN, [Residual Dense Network for Image Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Residual_Dense_Network_CVPR_2018_paper.html), **CVPR2018** Spotlight, [local and global Residual, bicubic downsampling, gaussian kernel feature fusing]  
[official-codes](https://github.com/yulunzhang/RDN)


- DBPN, [Deep Back-Projection Networks For Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2018/html/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.html), **CVPR2018**, [repeat down and up sample a back mechanism, Back-Projection]  
[pytorch-codes](https://github.com/alterzero/DBPN-Pytorch)


- ZSSR, ["Zero-Shot" Super-Resolution using Deep Internal Learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html), **CVPR2018**, [re-sample train test, internally train, zero-shot]  
[pytorch-codes](https://github.com/jacobgil/pytorch-zssr)


- SFTGAN, [Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html), **CVPR2018**, [semantic probability, semantic SFT]  
[pytorch-codes](https://github.com/xinntao/CVPR18-SFTGAN)


- EUSR, [Deep Residual Network with Enhanced Upscaling Module for Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Kim_Deep_Residual_Network_CVPR_2018_paper.html), **CVPR2018**, [enhanced upscaling module (EUM), change EDSR to EUSR by adding EUM]  
\-\-\-


- CARN, [Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network](), **ECCV2018**, [fast, cascading block]  
[pytorch-codes](https://github.com/nmhkahn/CARN-pytorch)


- GAN_degradation, [To learn image super-resolution, use a GAN to learn how to do image degradation first](http://openaccess.thecvf.com/content_ECCV_2018/html/Adrian_Bulat_To_learn_image_ECCV_2018_paper.html), **ECCV2018**, [mainly face test, use GAN to prodecu LR near to nature,]  
\-\-\-


- RCAN, [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](http://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html), **ECCV2018**, [Deep, Residual, Channel Attention, very deep residual block with channel attention using several skip connection and channel weight]  
[pytorch-codes](https://github.com/yulunzhang/RCAN)


- EPSR, [Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network](), **ECCV2018**, [has a new metrics idea]  
\-\-\-


##### In 2017
- DRRN, [Image Super-Resolution via Deep Recursive Residual Network](http://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html), **CVPR2017**, [residual network, combine ResNet and recursive]  
[caffe-codes](https://github.com/tyshiwo/DRRN_CVPR17)


- LapSRN, [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2017/html/Lai_Deep_Laplacian_Pyramid_CVPR_2017_paper.html), **CVPR2017**, [Pyramid network new loss to constrain]  
[matconvnet-codes](https://github.com/phoenix104104/LapSRN) | [pytorch](https://github.com/twtygqyy/pytorch-LapSRN) | [tensorflow](https://github.com/zjuela/LapSRN-tensorflow)


- SRDenseNet, [Image Super-Resolution Using Dense Skip Connections](http://openaccess.thecvf.com/content_iccv_2017/html/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.html), **ICCV2017**, [add dense block to model]  
[pytorch-codes](https://github.com/wxywhu/SRDenseNet-pytorch)


- SRGAN, [Photo-Realistic Single
