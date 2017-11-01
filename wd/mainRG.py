"""The registration module contains classes for image registration.

Image registration aims to align two images using a particular transformation.
miapy currently supports multi-modal rigid registration, i.e. align two images of different modalities
using a rigid transformation (rotation, translation, reflection, or their combination).

See Also:
    - `ITK Registration <https://itk.org/Doxygen/html/RegistrationPage.html>`_
    - `ITK Software Guide Registration <https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html>`_
"""
from enum import Enum
import matplotlib
matplotlib.use('Agg')  # use matplotlib without having a window appear
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import time

import mialab.utilities.pipeline_utilities as putil
import mialab.filtering.filter as fltr
import wd.registration.registration as R

d3D = True

# initialize evaluator
evaluator = putil.init_evaluator('./experiment1/')

if(d3D):
    #Testing 3D
    dimensions = 3
    loadTransformation = False

    # Read in the images:
    fixed_image = sitk.ReadImage('./atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz')
    moving_image = sitk.ReadImage('./test/100307/T1native.nii.gz')
    labels_native_image = sitk.ReadImage('./test/100307/labels_native.nii.gz')
    labels_mni_atlas = sitk.ReadImage('./test/100307/labels_mniatlas.nii.gz')

    # Define registration method:
    registration = R.MultiModalRegistration()  # specify parameters to your needs
    parameters = R.MultiModalRegistrationParams(fixed_image)

    # CREATE NEW TRANSFORMATION:
    if not loadTransformation:
        # Register the moving image and create the corresponding transformation during execute:
        start = time.time()
        registered_image = registration.execute(moving_image, parameters)
        exec_time = time.time() - start
        print('Total exection time: {}s'.format(exec_time))

        # Save transformaiton:
        sitk.WriteTransform(registration.transform, 'myTransformation.tfm')

    else:
        registration.transform = sitk.ReadTransform('myTransformation.tfm')
        # Apply the transformation to the moving image:
        registered_image = sitk.Resample(moving_image, registration.transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelIDValue())

    # Apply the transformation to the native lables image:
    labels_registred = sitk.Resample(labels_native_image, registration.transform, sitk.sitkLinear, 0.0, labels_native_image.GetPixelIDValue())

    # Subtract the registerd labels to get the error between the two images:
    subtracted_image = sitk.Subtract(labels_registred, labels_mni_atlas) #labels_registred - labels_mni_atlas;

    # Evaluate transformation:
    evaluator.evaluate(labels_registred,labels_mni_atlas,'eval_result')

    if exec_time > 0:
        file = open('./experiment1/results.csv', 'a')
        file.write('Total exection time; {}s'.format(exec_time))
        file.close()

    # Save the images:
    sitk.WriteImage(registered_image, 'myRegistred2.nii.gz')
    sitk.WriteImage(labels_registred, 'myRegistred_labels.nii.gz')
    sitk.WriteImage(subtracted_image, 'mySubtracted_labels.nii.gz')

else:
    #testing 2D
    dimensions = 2
    fixed_image = sitk.ReadImage('./DummyImages/RegistrationAmitAlpha.tif')
    moving_image = sitk.ReadImage('./DummyImages/RegistrationBmitAlpha.tif')
    registration = R.MultiModalRegistration()  # specify parameters to your needs
    parameters = R.MultiModalRegistrationParams(fixed_image)
    registered_image = registration.execute(moving_image, parameters)
    sitk.WriteImage(registered_image, 'DummyRegistred4.tif')
