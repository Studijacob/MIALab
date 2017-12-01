from __future__ import print_function

import SimpleITK as sitk
import sys
import os


def command_iteration(method) :
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))
    print("\t#: ", len(method.GetOptimizerPosition()))


def command_multi_iteration(method) :
    print("--------- Resolution Changing ---------")


# if len ( sys.argv ) < 4:
#     print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
#     sys.exit ( 1 )


fixed = sitk.ReadImage('../data/atlas/mni_icbm152_t2_tal_nlin_sym_09a.nii.gz', sitk.sitkFloat32)

moving = sitk.ReadImage('../data/test/899885/T1native.nii.gz', sitk.sitkFloat32)

# Set Basis Spline Mesh size in all 3 dimensions: x, y, z
transformDomainMeshSize = [14, 10, 12]

# Create BsplineTransform class from fixed-image and defined mesh-size:
tx_BsplineTransform = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

# Print initial parameters of the created BsplineTransform class:
print("Initial Parameters:");
print(tx_BsplineTransform.GetParameters())

# Create our Image-Registration-Method:
R = sitk.ImageRegistrationMethod()

# Set the metric used in our Image-Registration-Method:
R.SetMetricAsMattesMutualInformation(50)

# Set the Optimizer used in our Image-Registration-Method:
R.SetOptimizerAsGradientDescentLineSearch(5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5)

# Set the optimizer scales used in our Image-Registration-Method:
R.SetOptimizerScalesFromPhysicalShift()

# Set the initial transform used in our Image-Registration-Method:
R.SetInitialTransform(tx_BsplineTransform)

# Set the interpolator used in our Image-Registration-Method:
R.SetInterpolator(sitk.sitkLinear)

# Set the shrink-factors used in our Image-Registration-Method:
R.SetShrinkFactorsPerLevel([6, 3, 1])

# Set the smoothing-sigmas for each shrink-factor level used in our Image-Registration-Method:
R.SetSmoothingSigmasPerLevel([6, 3, 1])

# Add a command that gets called at every iteration during the execution of our Image-Registration-Method:
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

# Add a command that gets called at every resolution-change during the execution of our Image-Registration-Method:
R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R))

# Execute our Image-Registration-Method with the given images and save the transformation:
outTx = R.Execute(fixed, moving)

# Print the transformation and the optimizer stopping info:
print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

# Save the transformation:
sitk.WriteTransform(outTx, './BsplineResults/transform.tfm')

# if ( not "SITK_NOSHOW" in os.environ ):
#
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed);
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(100)
#     resampler.SetTransform(outTx)
#
#     out = resampler.Execute(moving)
#
#     # Save the images:
#     sitk.WriteImage(out, './BsplineResults/myOut.nii.gz')
#
#     #simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
#     #simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#     #cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
#     #sitk.Show( cimg, "ImageRegistration1 Composition" )