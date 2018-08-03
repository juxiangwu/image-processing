#coding:utf-8
import vtk
sphere = vtk.vtkSphereSource()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

actor1 = vtk.vtkActor()
actor1.SetMapper(mapper)
actor1.SetPosition(-0.2, -0.2, -0.6)
actor1.GetProperty().SetColor(0.8, 0.3, 0.5)

actor2 = vtk.vtkActor()
actor2.SetMapper(mapper)
actor2.SetPosition(0.2, 0.2, 0.6)
actor2.GetProperty().SetColor(0.1, 0.8, 0.5)
actor2.GetProperty().SetOpacity(0.2)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor1)
renderer.AddActor(actor2)
renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.SetSize(600,600)
window.AddRenderer(renderer)

window.Render()

iren=vtk.vtkRenderWindowInteractor() 
iren.SetRenderWindow(window)  
iren.Initialize() 
iren.Start()