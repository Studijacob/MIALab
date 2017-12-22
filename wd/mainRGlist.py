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
import numpy as np

import registration.registration as R
import registration.evalor as E

# initialize evaluator
evaluator = E.evalor()

timestamp = time.strftime("%Y%m%d-%H%M%S")
path = './experiment1/results'+str(timestamp)+'.csv'

# start the csv
file = open(path, 'w')
file.write(
    'ID; WhiteMatter; GreyMatter; Ventricles; Time; PatientID; Histogram; LearningRate; StepSize; Iteration; Shrinking; Smoothing;' + "\n")
file.close

# big job
PatientIDList = [899885, 188347, 189450, 190031, 192540, 196750, 198451, 199655, 201111, 208226]
histList = [100, 200, 1000]
learnRateList = [0.1, 0.2, 0.9]
stepSizeList = [0.001, 0.01, 0.1]
iterList = [100, 200, 1000]
shrinkList = [(2, 1, 1), (4, 2, 1), (8, 4, 1)]
smoothList = [(2, 1, 1), (4, 2, 1), (8, 4, 1)]

# small job
# PatientIDList = [899885, 188347]
# histList = [100]
# stepSizeList = [0.001]
# iterList = [100]
# shrinkList = [(2, 1, 1)]
# smoothList = [(2, 1, 1)]

for number_of_histogram_bins in histList:
    for learning_rate in learnRateList:
        for step_size in stepSizeList:
            for number_of_iterations in iterList:
                for shrink_factors in shrinkList:
                    for smoothing_sigmas in smoothList:
                        print("---")
                        i = 1
                        for patientID in PatientIDList:
                            print("patient:", patientID,
                                  "number_of_histogram_bins:", number_of_histogram_bins,
                                  "learning_rate:", learning_rate,
                                  "step_size:", step_size,
                                  "number_of_iterations:", number_of_iterations,
                                  "shrink_factors:", shrink_factors,
                                  "smoothing_sigmas: ", smoothing_sigmas, end="")
                            # Read in the images:
                            fixed_image = sitk.ReadImage('../data/test/' + str(patientID) + '/T1mni.nii.gz')
                            moving_image = sitk.ReadImage('../data/test/' + str(patientID) + '/T1native.nii.gz')
                            labels_native_image = sitk.ReadImage('../data/test/' + str(patientID) + '/labels_native.nii.gz')
                            labels_mni_atlas = sitk.ReadImage('../data/test/' + str(patientID) + '/labels_mniatlas.nii.gz')

                            registrationM = R.MultiModalRegistration(number_of_histogram_bins=number_of_histogram_bins,
                                                                     learning_rate=learning_rate,
                                                                     step_size=step_size,
                                                                     number_of_iterations=number_of_iterations,
                                                                     relaxation_factor=0.5,
                                                                     shrink_factors=shrink_factors,
                                                                     smoothing_sigmas=smoothing_sigmas,
                                                                     sampling_percentage=0.2)
                            parametersM = R.MultiModalRegistrationParams(fixed_image)

                            # Register the moving image and create the corresponding transformation during execute:
                            start = time.time()
                            registered_multi = registrationM.execute(moving_image, parametersM)
                            exec_time_m = time.time() - start

                            # Evaluate transformation:
                            # Apply the transformation to the native lables image:
                            labels_registredM = sitk.Resample(labels_native_image, registrationM.transform, sitk.sitkNearestNeighbor,
                                                              0.0,
                                                              labels_native_image.GetPixelIDValue())
                            resultsM = evaluator.evaluate(labels_registredM, labels_mni_atlas)

                            # write to result csv
                            resultsM = [i] + resultsM
                            if 'exec_time_m' in locals(): resultsM.append(exec_time_m)
                            print(" time:", exec_time_m)
                            resultsM.append(patientID)
                            resultsM.append(number_of_histogram_bins)
                            resultsM.append(learning_rate)
                            resultsM.append(step_size)
                            resultsM.append(number_of_iterations)
                            resultsM.append(shrink_factors)
                            resultsM.append(smoothing_sigmas)
                            file = open(path, "a")
                            writer = csv.writer(file, delimiter=';', lineterminator=';\r\n')
                            writer.writerow(resultsM)
                            file.close()
                            i += 1
