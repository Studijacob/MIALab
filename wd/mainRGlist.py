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
import SimpleITK as sitk
import time
import csv

import registration.registration as R
import registration.evalor as E


class RegistrationType(Enum):
    """Represents the registration transformation type."""
    AFFINE = 1
    RIGID = 2


# initialize evaluator
evaluator = E.evalor()


path = './experiment1/results.csv'

# start the csv
file = open(path, 'w')
file.write('WhiteMatter; GreyMatter; Ventricles; PatientID; Time;' + "\n")
file.close

PatientIDList = [899885, 188347, 189450, 190031, 192540, 196750, 198451, 199655, 201111, 208226]
# PatientIDList = [899885, 188347]

for patientID in PatientIDList:

    # Read in the images:
    print("PatientID:", patientID)
    print("load images ...", end="")
    #fixed_image = sitk.ReadImage('../data/atlas/mni_icbm152_t2_tal_nlin_sym_09a.nii.gz')
    fixed_image = sitk.ReadImage('../data/test/'+str(patientID)+'/T1mni.nii.gz')
    moving_image = sitk.ReadImage('../data/test/'+str(patientID)+'/T1native.nii.gz')
    labels_native_image = sitk.ReadImage('../data/test/'+str(patientID)+'/labels_native.nii.gz')
    labels_mni_atlas = sitk.ReadImage('../data/test/'+str(patientID)+'/labels_mniatlas.nii.gz')
    print(" done")

    # the parameters
    my_number_of_histogram_bins = 200  # int
    my_learning_rate = 0.10  # float
    my_step_size = 0.001  # float
    my_number_of_iterations = 200  # int
    my_relaxation_factor = 0.5  # int
    my_shrink_factors = (2, 1, 1)  # [int]
    my_smoothing_sigmas = (2, 1, 0)  # [float]
    my_sampling_percentage = 0.2  # float

    print("initialize multimodal transformation ... ", end="")
    registrationM = R.MultiModalRegistration(number_of_histogram_bins=my_number_of_histogram_bins,
                                            learning_rate=my_learning_rate,
                                            step_size=my_step_size,
                                            number_of_iterations=my_number_of_iterations,
                                            relaxation_factor=my_relaxation_factor,
                                            shrink_factors=my_shrink_factors,
                                            smoothing_sigmas=my_smoothing_sigmas,
                                            sampling_percentage=my_sampling_percentage)
    parametersM = R.MultiModalRegistrationParams(fixed_image)
    print("done")

    print("calculate affine transformation ...", end="", flush=True)
    # Register the moving image and create the corresponding transformation during execute:
    start = time.time()
    registered_multi = registrationM.execute(moving_image, parametersM)
    exec_time_m = time.time() - start
    print(" done")

    # Evaluate transformation:
    print("evaluating ... ", end="")
    # Apply the transformation to the native lables image:
    resultsA = evaluator.evaluate(labels_native_image, labels_mni_atlas)

    labels_registredM = sitk.Resample(labels_native_image, registrationM.transform, sitk.sitkLinear, 0.0,
                                      labels_native_image.GetPixelIDValue())
    resultsM = evaluator.evaluate(labels_registredM, labels_mni_atlas)

    print("done")
    print("Begin: ",resultsA)
    print("affine: ",resultsM)

    # write to result csv
    print("write results to file ... ", end="")
    resultsA.append(patientID)
    resultsM.append(patientID)
    if 'exec_time_m' in locals(): resultsM.append(exec_time_m)
    file = open(path, "a")
    writer = csv.writer(file, delimiter=';')
    writer.writerow(resultsA)
    writer.writerow(resultsM)
    file.close()
    print("done")
