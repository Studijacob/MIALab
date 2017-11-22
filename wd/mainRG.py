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
import csv
import os.path

import mialab.utilities.pipeline_utilities as putil
import mialab.filtering.filter as fltr
import registration.registration as R
import registration.evalor as E

class RegistrationType(Enum):
    """Represents the registration transformation type."""
    AFFINE = 1
    RIGID = 2

d3D = True

# initialize evaluator
evaluator = E.evalor()

if(d3D):
    #Testing 3D
    dimensions = 3
    loadTransformation = False
    PatientIDList = [100307, 188347, 189450, 190031, 192540, 196750, 198451, 199655, 201111, 208226]
    patientID = 100307
    path = './experiment1/results.csv'

    # start the csv
    file = open(path, 'w')
    file.write('WhiteMatter; GreyMatter; Ventricles; PatientID; Time;' + "\n")
    file.close

    # Read in the images:
    print("load images ...", end="")
    fixed_image = sitk.ReadImage('../data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz')

    moving_image = sitk.ReadImage('../data/test/100307/T1native.nii.gz')

    labels_native_image = sitk.ReadImage('../data/test/100307/labels_native.nii.gz')
    labels_mni_atlas = sitk.ReadImage('../data/test/100307/labels_mniatlas.nii.gz')
    print(" done")

    # Do several registrations:
    # nhistogramBins = [10, 50, 100, 150, 200, 250, 300, 400] # many different bin size
    nhistogramBins = [200] # default bin size
    for i in nhistogramBins:
        mode = "bspline" #bspline , multimodal

        print("initialize transformation ... ", end="")
        if mode == "multimodal":
            # Define registration method:
            my_registration_type = RegistrationType.AFFINE
            my_number_of_histogram_bins = i # int
            my_learning_rate = 1.0 # float
            my_step_size = 0.001 # float
            my_number_of_iterations = 200 # int
            my_relaxation_factor = 0.5 # int
            my_shrink_factors = (2, 1, 1) # [int]
            my_smoothing_sigmas = (2, 1, 0) # [float]
            my_sampling_percentage = 0.2 # float

            registration = R.MultiModalRegistration(number_of_histogram_bins=my_number_of_histogram_bins,
                                                    learning_rate=my_learning_rate,
                                                    step_size=my_step_size,
                                                    number_of_iterations=my_number_of_iterations,
                                                    relaxation_factor=my_relaxation_factor,
                                                    shrink_factors=my_shrink_factors,
                                                    smoothing_sigmas=my_smoothing_sigmas,
                                                    sampling_percentage=my_sampling_percentage)  # specify parameters to your needs
            parameters = R.MultiModalRegistrationParams(fixed_image)
        elif mode == "bspline":
            registration = R.BSplineRegistration()
            parameters = R.BSplineRegistrationParams(fixed_image)
        else:
            print("no correct Model")
        print("done")

        print("calculate transformation ...", end="", flush=True)
        # CREATE NEW TRANSFORMATION:
        if loadTransformation and os.path.isfile('./myTransformation.tfm'):
            registration.transform = sitk.ReadTransform('./myTransformation.tfm')
            # Apply the transformation to the moving image:
            registered_image = sitk.Resample(moving_image, registration.transform, sitk.sitkLinear, 0.0,
                                             moving_image.GetPixelIDValue())
        else:
            # Register the moving image and create the corresponding transformation during execute:
            start = time.time()
            registered_image = registration.execute(moving_image, parameters)
            exec_time = time.time() - start

            print('Total exection time: {}'.format(exec_time))

            # Save transformaiton:
            sitk.WriteTransform(registration.transform, './myTransformation.tfm')


        print(" done")

        # Apply the transformation to the native lables image:
        labels_registred = sitk.Resample(labels_native_image, registration.transform, sitk.sitkLinear, 0.0, labels_native_image.GetPixelIDValue())

        # Evaluate transformation:
        print("evaluating ... ", end="")
        # results = evaluator.evaluate(labels_registred,labels_mni_atlas)
        results = evaluator.evaluate(labels_registred,labels_mni_atlas)
        print("done")

        sitk.WriteImage(registered_image, 'myRegistred2.nii.gz')

        # write to file
        print("write results to file ... ", end="")
        results.append(patientID)
        if 'exec_time' in locals(): results.append(exec_time)
        file = open(path, "a")
        writer = csv.writer(file, delimiter=';')
        writer.writerow(results)
        file.close()
        print("done")

        # Save the images:

        # sitk.WriteImage(labels_registred, 'myRegistred_labels.nii.gz')
        # sitk.WriteImage(subtracted_image, 'mySubtracted_labels.nii.gz')

else:
    #testing 2D
    dimensions = 2
    fixed_image = sitk.ReadImage('./DummyImages/RegistrationAmitAlpha.tif')
    moving_image = sitk.ReadImage('./DummyImages/RegistrationBmitAlpha.tif')
    registration = R.MultiModalRegistration()  # specify parameters to your needs
    parameters = R.MultiModalRegistrationParams(fixed_image)
    registered_image = registration.execute(moving_image, parameters)
    sitk.WriteImage(registered_image, 'DummyRegistred4.tif')
