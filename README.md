# GAN-for-CT-Image-generation
Repository contains software which is able to create new CT images from a new CT image. The repository contains a Generative Adversarial Network in the project.py file. This network's architecture obeys object oriented architecture. More information can be found in "PGM_Progress Report".

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

FUTURE GOALS (in implementation priority)
- Make test output to be easier to check with original image
- Automated BRISQUE test of images
- PSNR comparison for close looking images
- Create github instructions
- Better in code documentation

CITATIONS:
Bousmalis, K., et all. Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Network. 2017 IEEE Conference on Computer Vision and Patter Recognition (CVPR) 95-104.

Linder-Noren, E. Keras-GAN. 2017. GitHub. https://github.com/eriklindernoren/Keras-GAN

Chollet, F., et all. Keras: The Python Deep Learning Library, Keras Documentation, 2018.
    
Humphries, T., Si, D.Coulter, S., Simms, M., Xing, R. (1 March 2019). Comparison of deep learning approaches to low dose CT using low intensity and sparse view data. Medical Imaging 2019: Physics of Medical Imaging. DOI 10.1117.
    
Clark, k. et all. The Cancer Imaging Archieve(TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.
    
Goncalves Mokarzel, P. GAN for CT Image Generation. GitHub. https://github.com/pgmoka/GAN-for-CT-Image-generation
