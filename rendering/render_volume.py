from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'dataset/vortex.raw'
SHAPE = (128, 128, 128)  # 尺寸顺序为 (Nx, Ny, Nz)


def make_volume(path, shape):
    # raw 中的 x 索引变化最快，与 VTK 的标量排列顺序一致。
    values = np.fromfile(path, dtype='<f4')
    image = vtk.vtkImageData()
    image.SetDimensions(*shape)
    image.SetOrigin(-1, -1, -1)
    image.SetSpacing(*(2 / n for n in shape))
    image.GetPointData().SetScalars(numpy_to_vtk(values, deep=True))

    # 将标量值映射为颜色和不透明度。
    low, high = image.GetScalarRange()
    span = high - low
    colors = vtk.vtkColorTransferFunction()
    # VolumeVisual 的 Vortex 配色；位置按数据范围归一化到 [0,1]。
    colors.SetColorSpaceToLab()
    for position, r, g, b in [
        (0.000000, 0, 0, 255),
        (0.351562, 0, 170, 255),
        (0.507812, 255, 41, 2),
        (0.656250, 255, 238, 17),
        (1.000000, 219, 219, 219),
    ]:
        colors.AddRGBPoint(low + position * span, r / 255, g / 255, b / 255)

    opacity = vtk.vtkPiecewiseFunction()
    # 两个不透明度峰突出不同数值区间，其余区域保持透明。
    for position, alpha in [
        (0.000000, 0.000000),
        (0.318359, 0.000000),
        (0.332031, 0.144000),
        (0.359375, 0.210667),
        (0.380859, 0.150667),
        (0.390625, 0.000000),
        (0.445312, 0.000000),
        (0.457031, 0.257333),
        (0.466797, 0.410667),
        (0.482422, 0.584000),
        (0.513672, 0.644000),
        (0.539062, 0.557333),
        (0.552734, 0.410667),
        (0.558594, 0.277333),
        (0.566406, 0.000000),
        (1.000000, 0.000000),
    ]:
        opacity.AddPoint(low + position * span, alpha)

    properties = vtk.vtkVolumeProperty()
    properties.SetColor(colors)
    properties.SetScalarOpacity(opacity)
    properties.SetScalarOpacityUnitDistance(2 / 64)
    properties.SetInterpolationTypeToLinear()
    properties.ShadeOn()
    properties.SetAmbient(0.3)
    properties.SetDiffuse(0.7)
    properties.SetSpecular(0.2)

    # 使用 GPU 光线投射，沿射线合成颜色和不透明度。
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(image)
    mapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(properties)
    return volume


def main():
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(make_volume(RAW_PATH, SHAPE))
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
    window.SetWindowName('VTK Volume Rendering')

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
