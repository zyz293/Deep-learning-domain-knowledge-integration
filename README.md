# Deep learning based domain knowledge integration for small datasets: illustrative applications in materials informatics

## Requirements ##
Python 3.6.3; 
Numpy 1.18.1; 
Sklearn 0.20.0; 
Keras 2.3.1; 
Pickle 4.0; 
TensorFlow 2.1.0; 
h5py 2.9.0;

## Files ##
1. SDM_MURI_homogenization.py: The CNN has two branches where one branch takes the original 3D microstructure image as input and the other brach takes the corresponding two-point autocorrelation as input. Finally the two branches are concatenated togetogher to make the final prediction for effective elastic stiffness. The details please refer to the paper in the reference section.
2. SDM_NIST_olddata.py: The CNN has two branches where one branch takes the original 2D microstructure image as input and the other brach takes the corresponding two-point autocorrelation as input. Finally the two branches are concatenated togetogher to make the final prediction for initial deformation level. The details please refer to the paper in the reference section.


## How to run it
1. To run SDM_MURI_homogenization.py: use commend 'python SDM_MURI_homogenization.py'. The script will train the CNN and save your CNN.
2. To run SDM_NIST_olddata.py: use commend 'python SDM_NIST_olddata.py'. The script will train the CNN and save your CNN.


## Acknowledgement
This work was performed under the following financial assistance award 70NANB14H012 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from the following grants: NSF award CCF-1409601; DOE awards DE-SC0007456, DE-SC0014330; AFOSR award FA9550-12-1-0458, and Northwestern Data Science Initiative. 

## Related Publications ##
Z. Yang, R. Al-Bahrani, A. Reid, S. Papanikolaou, S. Kalidindi, W.-keng Liao, A. Choudhary, and A. Agrawal, “Deep learning based domain knowledge integration for small datasets: Illustrative applications in materials informatics,” in Proceedings of International Joint Conference on Neural Networks (IJCNN), 2019, pp. 1–8.

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>
