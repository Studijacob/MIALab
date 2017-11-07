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

class RegistrationType(Enum):
    """Represents the registration transformation type."""
    AFFINE = 1
    RIGID = 2

d3D = True

# initialize evaluator
evaluator = putil.init_evaluator('./experiment1/')

if(d3D):
    #Testing 3D
    dimensions = 3
    loadTransformation = False

    # Read in the images:
    print("load images ...", end="")
    fixed_image = sitk.ReadImage('..data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz')
    moving_image = sitk.ReadImage('../data/test/100307/T1native.nii.gz')
    labels_native_image = sitk.ReadImage('../data/test/100307/labels_native.nii.gz')
    labels_mni_atlas = sitk.ReadImage('../data/test/100307/labels_mniatlas.nii.gz')
    print(" done")

    file = open('./experiment1/results.csv', 'a')
    file.write('Patient Nr;' + ' 100307' + "\n")
    file.close

# Do several registrations:
    nhistogramBins = [10, 50, 100, 150, 200, 250, 300, 400]
    for i in nhistogramBins:
        # Define registration method:
        print("initialize transformation ... ", end="")
        my_registration_type = RegistrationType.AFFINE
        my_number_of_histogram_bins = i # int
        my_learning_rate = 1.0 # float
        my_step_size = 0.001 # float
        my_number_of_iterations = 200 # int
        my_relaxation_factor = 0.5 # int
        my_shrink_factors = (2, 1, 1) # [int]
        my_smoothing_sigmas = (2, 1, 0) # [float]
        my_sampling_percentage = 0.2 # float

        registration = R.MultiModalRegistration(number_of_histogram_bins=my_number_of_histogram_bins, learning_rate=my_learning_rate, step_size=my_step_size, number_of_iterations=my_number_of_iterations, relaxation_factor=my_relaxation_factor, shrink_factors=my_shrink_factors, smoothing_sigmas=my_smoothing_sigmas, sampling_percentage=my_sampling_percentage)  # specify parameters to your needs
        parameters = R.MultiModalRegistrationParams(fixed_image)
        print("done")

        # CREATE NEW TRANSFORMATION:
        if not loadTransformation:
            # Register the moving image and create the corresponding transformation during execute:
            print("calculate transformation ...", end="")
            start = time.time()
            registered_image = registration.execute(moving_image, parameters)
            exec_time = time.time() - start
            print(" done")
            print('Total exection time: {}'.format(exec_time))

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
        print("evaluating ...")
        evaluator.evaluate(labels_registred,labels_mni_atlas,'eval_result')

        if exec_time > 0:
            file = open('./experiment1/results.csv', 'a')
            file.write('Total exection time; {}'.format(exec_time) + '\n')
            file.close()

    # Save the images:
    # sitk.WriteImage(registered_image, 'myRegistred2.nii.gz')
    # sitk.WriteImage(labels_registred, 'myRegistred_labels.nii.gz')
    # sitk.WriteImage(subtracted_image, 'mySubtracted_labels.nii.gz')

    # https://stackoverflow.com/questions/5598181/python-print-on-same-line
    # https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder

else:
    #testing 2D
    dimensions = 2
    fixed_image = sitk.ReadImage('./DummyImages/RegistrationAmitAlpha.tif')
    moving_image = sitk.ReadImage('./DummyImages/RegistrationBmitAlpha.tif')
    registration = R.MultiModalRegistration()  # specify parameters to your needs
    parameters = R.MultiModalRegistrationParams(fixed_image)
    registered_image = registration.execute(moving_image, parameters)
    sitk.WriteImage(registered_image, 'DummyRegistred4.tif')
