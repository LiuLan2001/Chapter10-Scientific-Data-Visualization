from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'dataset/vortex.raw'
SHAPE = (128, 128, 128)
ISOVALUE = 0.1  # 数据归一化到 [0,1] 后的等值阈值
SURFACE_COLOR = (0.80, 0.80, 0.64)  # 浅黄橄榄色


def make_isosurface(path, shape, isovalue):
    values = np.fromfile(path, dtype='<f4')
    values = (values - values.min()) / (values.max() - values.min())
    image = vtk.vtkImageData()
    image.SetDimensions(*shape)
    image.SetOrigin(-1, -1, -1)
    image.SetSpacing(*(2 / n for n in shape))
    image.GetPointData().SetScalars(numpy_to_vtk(values, deep=True))

    # 提取给定阈值的等值面，并计算光照所需的法线。
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, isovalue)
    contour.ComputeNormalsOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    surface = vtk.vtkActor()
    surface.SetMapper(mapper)
    properties = surface.GetProperty()
    properties.SetColor(*SURFACE_COLOR)
    properties.SetInterpolationToPhong()
    properties.SetAmbient(0.25)
    properties.SetDiffuse(0.70)
    properties.SetSpecular(0.40)
    properties.SetSpecularPower(32)
    return surface


def main():
    renderer = vtk.vtkRenderer()
    renderer.AddActor(make_isosurface(RAW_PATH, SHAPE, ISOVALUE))
    renderer.SetBackground(1, 1, 1)

    # 光源跟随相机，从观察方向照向体数据。
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    light.SetColor(1, 1, 1)
    light.SetIntensity(1.0)
    renderer.AddLight(light)

    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(20)
    renderer.ResetCameraClippingRange()

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(900, 900)
    window.SetWindowName('VTK Isosurface Rendering')

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    # 左下角的坐标轴随观察方向转动。
    axes = vtk.vtkAxesActor()
    for caption in [axes.GetXAxisCaptionActor2D(),
                    axes.GetYAxisCaptionActor2D(),
                    axes.GetZAxisCaptionActor2D()]:
        text = caption.GetCaptionTextProperty()
        text.SetColor(0, 0, 0)
        text.SetFontFamilyToTimes()
        text.ShadowOff()
    orientation = vtk.vtkOrientationMarkerWidget()
    orientation.SetOrientationMarker(axes)
    orientation.SetInteractor(interactor)
    orientation.SetViewport(0.0, 0.0, 0.20, 0.20)
    orientation.SetEnabled(1)
    orientation.InteractiveOff()

    window.Render()
    interactor.Start()


if __name__ == '__main__':
    main()
