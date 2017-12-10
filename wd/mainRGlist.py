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

    print("initialize multimodal transformation ... ", end="")
    registrationM = R.MultiModalRegistration()
    parametersM = R.MultiModalRegistrationParams(fixed_image)
    print("done")

    print("initialize bspline transformation ... ", end="")
    registrationB = R.BSplineRegistration()
    parametersB = R.BSplineRegistrationParams(fixed_image)
    print("done")

    print("calculate affine transformation ...", end="", flush=True)

    # Register the moving image and create the corresponding transformation during execute:
    start = time.time()
    registered_multi = registrationM.execute(moving_image, parametersM)
    exec_time_m = time.time() - start
    print(" done")
    print("calculate bspline transformation ...", end="", flush=True)
    start = time.time()
    registered_b = registrationB.execute(registered_multi, parametersB)
    exec_time_b = time.time() - start
    print("done")

    print('Total exection time: {}'.format(exec_time_b))

    # Evaluate transformation:
    print("evaluating ... ", end="")
    # Apply the transformation to the native lables image:
    resultsA = evaluator.evaluate(labels_native_image, labels_mni_atlas)

    labels_registredM = sitk.Resample(labels_native_image, registrationM.transform, sitk.sitkLinear, 0.0,
                                      labels_native_image.GetPixelIDValue())
    resultsM = evaluator.evaluate(labels_registredM, labels_mni_atlas)

    labels_registredB = sitk.Resample(labels_registredM, registrationB.transform, sitk.sitkLinear, 0.0,
                                      labels_native_image.GetPixelIDValue())
    results = evaluator.evaluate(labels_registredB, labels_mni_atlas)

    print("done")
    print("Begin: ",resultsA)
    print("affine: ",resultsM)
    print("bspline:", results)

    # write to result csv
    print("write results to file ... ", end="")
    resultsA.append(patientID)
    resultsM.append(patientID)
    results.append(patientID)
    if 'exec_time_m' in locals(): resultsM.append(exec_time_m)
    if 'exec_time_b' in locals(): results.append(exec_time_b)
    file = open(path, "a")
    writer = csv.writer(file, delimiter=';')
    writer.writerow(resultsA)
    writer.writerow(resultsM)
    writer.writerow(results)
    file.close()
    print("done")

# Save the images:
# sitk.WriteImage(registered_multi, 'myRegistredM.nii.gz')
# sitk.WriteImage(registered_b, 'myRegistredB.nii.gz')
