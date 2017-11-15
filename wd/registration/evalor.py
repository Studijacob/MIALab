import mialab.evaluation.metric as mtrc
import numpy as np
import SimpleITK as sitk
from typing import Union

class evalor():
    def __init__(self):
        # init_evaluator
        self.metrics = []  # list of IMetrics
        self.labels = {}  # dictionary of label: label_str

        self.labels[1] = "WhiteMatter"
        self.labels[2] = "GreyMatter"
        self.labels[3] = "Ventricles"
        self.metrics.append(mtrc.DiceCoefficient())

    # evaluate
    def evaluate(self, image: Union[sitk.Image, np.ndarray], ground_truth: Union[sitk.Image, np.ndarray]):
        image_array = sitk.GetArrayFromImage(image) if isinstance(image, sitk.Image) else image
        ground_truth_array = sitk.GetArrayFromImage(ground_truth) if isinstance(ground_truth, sitk.Image) else ground_truth

        results = []

        for label, label_str in self.labels.items():
            # get only current label
            predictions = np.in1d(image_array.ravel(), label, True).reshape(image_array.shape).astype(np.uint8)
            labels = np.in1d(ground_truth_array.ravel(), label, True).reshape(ground_truth_array.shape).astype(np.uint8)

            # calculate the confusion matrix for IConfusionMatrixMetric
            confusion_matrix = mtrc.ConfusionMatrix(predictions, labels)

            # flag indicating whether the images have been converted for ISimpleITKImageMetric
            converted_to_image = False
            predictions_as_image = None
            labels_as_image = None

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, mtrc.IConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix
                elif isinstance(metric, mtrc.INumpyArrayMetric):
                    metric.ground_truth = labels
                    metric.segmentation = predictions
                elif isinstance(metric, mtrc.ISimpleITKImageMetric):
                    if not converted_to_image:
                        predictions_as_image = sitk.GetImageFromArray(predictions)
                        predictions_as_image.CopyInformation(image)
                        labels_as_image = sitk.GetImageFromArray(labels)
                        labels_as_image.CopyInformation(ground_truth)
                        converted_to_image = True

                    metric.ground_truth = labels_as_image
                    metric.segmentation = predictions_as_image

                results.append(metric.calculate())
        return results

