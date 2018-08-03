#coding:utf-8
import vtk

sph = vtk.vtkSphereSource()
sph.SetCenter(0.0,0.0,0.0)
sph.SetRadius(1.0)
sph.SetThetaResolution(64)
sph.SetPhiResolution(64)

m=vtk.vtkPolyDataMapper()
m.SetInputConnection(sph.GetOutputPort())

a=vtk.vtkActor()
a.SetMapper(m)
a.GetProperty().SetColor(0.0, 1.0, 0.0)
a.GetProperty().SetOpacity(1.0)
a.GetProperty().SetInterpolationToFlat()
a.SetOrientation(45.,0.,0.)

r=vtk.vtkRenderer()
r.AddActor(a)
r.ResetCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(r)
window.SetSize(800,800)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()

renderLarge = vtk.vtkRenderLargeImage()
renderLarge.SetInput(r)
renderLarge.SetMagnification(4)

png = vtk.vtkPNGWriter()
png.SetInputConnection(renderLarge.GetOutputPort())
png.SetFileName("temp/foo.png")
png.Write()
