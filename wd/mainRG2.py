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


# initialize evaluator
evaluator = E.evalor()

# Testing 3D
dimensions = 3
patientID = 899885
path = './experiment1/results.csv'

# start the csv
file = open(path, 'w')
file.write('WhiteMatter; GreyMatter; Ventricles; PatientID; Time;' + "\n")
file.close

# Read in the images:
print("load images ...", end="")
fixed_image = sitk.ReadImage('../data/test/208226/T1mni.nii.gz')
#fixed_image = sitk.ReadImage('../data/atlas/mni_icbm152_t2_tal_nlin_sym_09a.nii.gz')

moving_image = sitk.ReadImage('../data/test/899885/T1native.nii.gz')

labels_native_image = sitk.ReadImage('../data/test/899885/labels_native.nii.gz')
labels_mni_atlas = sitk.ReadImage('../data/test/208226/labels_mniatlas.nii.gz')
#labels_mni_atlas = sitk.ReadImage('../data/test/899885/labels_mniatlas.nii.gz')
print(" done")

print("initialize multimodal transformation ... ", end="")
# Define registration method:
my_registration_type = RegistrationType.AFFINE
my_number_of_histogram_bins = 200  # int
my_learning_rate = 0.10  # float
my_step_size = 0.001  # float
my_number_of_iterations = 200  # int
my_relaxation_factor = 0.5  # int
my_shrink_factors = (2, 1, 1)  # [int]
my_smoothing_sigmas = (2, 1, 0)  # [float]
my_sampling_percentage = 0.2  # float

registration = R.MultiModalRegistration(number_of_histogram_bins=my_number_of_histogram_bins,
                                        learning_rate=my_learning_rate,
                                        step_size=my_step_size,
                                        number_of_iterations=my_number_of_iterations,
                                        relaxation_factor=my_relaxation_factor,
                                        shrink_factors=my_shrink_factors,
                                        smoothing_sigmas=my_smoothing_sigmas,
                                        sampling_percentage=my_sampling_percentage)  # specify parameters to your needs
parameters = R.MultiModalRegistrationParams(fixed_image)
print("done")

print("initialize bspline transformation ... ", end="")
registrationB = R.BSplineRegistration()
parametersB = R.BSplineRegistrationParams(fixed_image)
print("done")

print("calculate Multi transformation ...", end="", flush=True)

# Register the moving image and create the corresponding transformation during execute:
start = time.time()
registered_multi = registration.execute(moving_image, parameters)
print(" done")
print("calculate bspline transformation ...", end="", flush=True)
registered_b = registrationB.execute(registered_multi, parametersB)
exec_time = time.time() - start
print("done")

print('Total exection time: {}'.format(exec_time))

# Save transformaiton:
sitk.WriteTransform(registration.transform, './Transformations/myTransformation.tfm')

# Apply the transformation to the native lables image:
labels_registred = sitk.Resample(labels_native_image, registration.transform, sitk.sitkLinear, 0.0,
                                 labels_native_image.GetPixelIDValue())

# Evaluate transformation:
print("evaluating ... ", end="")
# results = evaluator.evaluate(labels_registred,labels_mni_atlas)
results = evaluator.evaluate(labels_registred, labels_mni_atlas)
print("done")

# write to result csv
print("write results to file ... ", end="")
results.append(patientID)
if 'exec_time' in locals(): results.append(exec_time)
file = open(path, "a")
writer = csv.writer(file, delimiter=';')
writer.writerow(results)
file.close()
print("done")

# Save the images:
sitk.WriteImage(registered_multi, 'myRegistredM.nii.gz')
sitk.WriteImage(registered_b, 'myRegistredB.nii.gz')
