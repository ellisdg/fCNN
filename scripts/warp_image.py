#!/usr/bin/env python
import itk
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: ' + sys.argv[0] +
              ' <InputFileName> <DisplacementFieldFileName> <OutputFileName>')
        sys.exit(1)

    inputFileName = sys.argv[1]
    displacementFieldFileName = sys.argv[2]
    outputFileName = sys.argv[3]

    Dimension = 2

    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, Dimension]

    DisplacementFieldType = itk.Image[VectorPixelType, Dimension]

    PixelType = itk.UC
    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(inputFileName)

    fieldReader = itk.ImageFileReader[DisplacementFieldType].New()
    fieldReader.SetFileName(displacementFieldFileName)
    fieldReader.Update()

    deformationField = fieldReader.GetOutput()

    warpFilter = \
            itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()

    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

    warpFilter.SetInterpolator(interpolator)

    warpFilter.SetOutputSpacing(deformationField.GetSpacing())
    warpFilter.SetOutputOrigin(deformationField.GetOrigin())
    warpFilter.SetOutputDirection(deformationField.GetDirection())

    warpFilter.SetDisplacementField(deformationField)

    warpFilter.SetInput(reader.GetOutput())

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetInput(warpFilter.GetOutput())
    writer.SetFileName(outputFileName)

    writer.Update()
