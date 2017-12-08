import SimpleITK as sitk

elastixImageFilter = sitk.ElastixImageFilter()

fixed_image = sitk.ReadImage('../data/test/899885/T1mni.nii.gz')
#fixed_image = sitk.ReadImage('../data/atlas/mni_icbm152_t2_tal_nlin_sym_09a.nii.gz')
moving_image = sitk.ReadImage('../data/test/899885/T1native.nii.gz')

elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetMovingImage(moving_image)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)

elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage())