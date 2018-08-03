#coding:utf-8
import vtk
import random
import numpy

N = 200
pts = numpy.zeros((N,3))
for i in range(N):
    pts[i,0] = random.uniform(0, 1.0)
    pts[i,1] = random.uniform(0, 1.0)
    pts[i,2] = random.uniform(0, 1.0)

sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.05)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

renderer = vtk.vtkRenderer()
actors = []
for i in range(N):
    actors.append(vtk.vtkActor())
    actors[-1].SetMapper(mapper)
    actors[-1].SetPosition(pts[i])
    actors[-1].GetProperty().SetColor(pts[i,2], 1.0-pts[i,2], 0.0)
    renderer.AddActor(actors[-1])
#renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()
