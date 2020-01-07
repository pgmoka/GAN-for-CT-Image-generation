# GAN-for-CT-Image-generation
Repository contains software which is able to create new CT images from a new CT image. The repository contains a Generative Adversarial Network in the project.py file. This network's architecture obeys object oriented architecture. More information can be found in "Code Guide", "PGM_Progress Report", and "PGM_Final Paper".

CURRENT DEVELOPMENT PHASE
Model is able to successfully create CT_Images from input. Success parameters in implementation. Further execution structions found inside the 'software' directory.

INFORMATION
This project has been based in the research work for the paper in this repository("Image-to-ImageTranslationwithConditionalAdversarialNetworks.pdf"), and the Git repository of Keras implementations of Generative Adversarial Networks(https://github.com/eriklindernoren/Keras-GAN). 

ACHIEVED GOALS (in time implementation)
- Research GAN phase
- Class design
- Implementation of general flow
- Neural network initial implementation
    - Initial flow
    - Refactoring
- First successfull local test
- Create better, more relevant output
- Implement batch
- Migrate to stronger server to train with the full database
    - Create better input for remote machine
    - Create a guide for image creation
- Create a success standard for image creation
    - Create software to test this easily
- Make test output to be easier to check with original image
- Create github instructions
- Automated BRISQUE test of images
- PSNR comparison for close looking images
- Better in code documentation

CITATIONS:
[1]	Bousmalis, K., et all. 2017. Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Network. 2017 IEEE Conference on Computer Vision and Patter Recognition (CVPR) 95-104.
[2]	Linder-Noren, E. 2017. Keras-GAN. GitHub. https://github.com/eriklindernoren/Keras-GAN
[3] Chollet, F., et all. 2018. Keras: The Python Deep Learning Library, Keras Documentation.
[4]	Humphries, T., Si, D.Coulter, S., Simms, M., Xing, R. 1 March 2019. Comparison of deep learning approaches to low dose CT using low intensity and sparse view data. Medical Imaging 2019: Physics of Medical Imaging. DOI 10.1117.
[5]	Mittal, A., Moorthy, A., & Bovik, A. 2012. No-Reference Image Quality Assessment in the Spatial Domain. IEEE Transactions on Image Processing, 21(12), 4695-4708.
[6] Shrimali, K. R.. Image Quality Assessment: BRISQUE. June 3, 2018. Learn OpenCV. Retrieved from: www.learnopencv.com/image-quality-assessment-brisque/
[7]	MathWorks. 2019. Brisque. Documentation. Retrieved from www.mathworks.com/help/images/ref/brisque.html.
[8]	MathWorks. PSNR. 2019. Documentation . Retrieved from. www.mathworks.com/help/vision/ref/psnr.html
[9]	Peak Signal-to-noise Ratio. (n.d.). Retrieved from en.wikipedia.org/wiki/Peak_signal-to-noise_ratio.
[10] Clark, k. et all. The Cancer Imaging Archieve(TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.
[11] Karras, T., Laine, S., & Aila, T. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks.

    
Goncalves Mokarzel, P. GAN for CT Image Generation. GitHub. https://github.com/pgmoka/GAN-for-CT-Image-generation
