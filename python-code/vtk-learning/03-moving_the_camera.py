#coding:utf-8
import vtk
import time

cone = vtk.vtkConeSource()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cone.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.ResetCamera()
camera = renderer.GetActiveCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

azimuth = 0
while 1:
	window.Render()
	if azimuth >= 360:
		azimuth = 0
	azimuth += 0.1
	camera.Azimuth(azimuth)

	time.sleep(0.1)