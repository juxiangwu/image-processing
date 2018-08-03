#coding:utf-8
import vtk

cone = vtk.vtkConeSource()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cone.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.SetPosition(0, 0, 0)
actor.GetProperty().SetColor(0.8, 0.6, 0.1)
actor.GetProperty().SetAmbient(0.1)
actor.GetProperty().SetDiffuse(0.6)
actor.GetProperty().SetSpecular(0.8)

camera = vtk.vtkCamera()
camera.SetFocalPoint(0.0, 0.0, 0.0)
camera.SetPosition(0.0, 0.0, 8.0)
camera.SetViewUp(0.0, 1.0, 0.0)

light = vtk.vtkLight()
light.SetFocalPoint(0.0, 0.0, 0.0)
light.SetPosition(2.0, 2.0, 10.0)
light.SetConeAngle(100.0)
light.SetLightTypeToCameraLight()

spot = vtk.vtkLight()
spot.SetFocalPoint(0.0, 0.0, 0.0)
spot.SetPosition(-6.0, -10.0, 4.0)
spot.SetConeAngle(10.0)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.AddLight(light)
renderer.AddLight(spot)
renderer.SetActiveCamera(camera)
renderer.LightFollowCameraOn()

window = vtk.vtkRenderWindow()
window.SetSize(800,600)
window.AddRenderer(renderer)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()