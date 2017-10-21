from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import sys
import os
import time

def command_iteration(method) :
     print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                    method.GetMetricValue(),
                                    method.GetOptimizerPosition()))



def RSGD():
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    return R


def GDLS(fixed, moving):
    fixed = sitk.Normalize(fixed)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)
    moving = sitk.Normalize(moving)
    moving = sitk.DiscreteGaussian(moving, 2.0)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                              numberOfIterations=200,
                                              convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    return fixed, moving, R


def corr_RSGD():
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                               minStep=1e-4,
                                               numberOfIterations=500,
                                               gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    return R


def MMI_RSGD():
    numberOfBins = 24
    samplingPercentage = 0.10
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    return R


fixed = sitk.ReadImage('./test/100307/T1mni.nii.gz', sitk.sitkFloat32)
moving = sitk.ReadImage('./test/100307/T2mni.nii.gz', sitk.sitkFloat32)

# different registration and metric systems
# R = MMI_RSGD()
# R = corr_RSGD()
# fixed, moving, R = GDLS()
R = RSGD()

R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

# We'll meausure the execution time
start = time.time()

outTx = R.Execute(fixed, moving)

exec_time = time.time()-start
print('Total exection time: {}s'.format(exec_time))

print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

sitk.WriteTransform(outTx,  'myRegistred.nii.gz')

# if ( not "SITK_NOSHOW" in os.environ ):
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(100)