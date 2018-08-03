#coding:utf-8
import vtk
import os
curdir,fname = os.path.split(__file__)
model_path = os.path.join(curdir,'models/crown.den.cube')
density_neutral = vtk.vtkGaussianCubeReader()
density_neutral.SetFileName(os.path.join(curdir,"models/crown.den.cube"))
density_neutral.Update()

potential_cation = vtk.vtkGaussianCubeReader()
potential_cation.SetFileName(os.path.join(curdir,"models/crownK.pot.cube"))
potential_cation.Update()

potential_neutral = vtk.vtkGaussianCubeReader()
potential_neutral.SetFileName(os.path.join(curdir,"models/crown.pot.cube"))
potential_neutral.Update()

contour = vtk.vtkContourFilter()
contour.SetInputData(density_neutral.GetGridOutput())
contour.SetValue(0, 0.05)

potential_difference = vtk.vtkImageMathematics()
potential_difference.SetOperationToSubtract()
potential_difference.SetInput1Data(potential_cation.GetGridOutput())
potential_difference.SetInput2Data(potential_neutral.GetGridOutput())

probe = vtk.vtkProbeFilter()
probe.SetInputConnection(contour.GetOutputPort())
probe.SetSourceConnection(potential_difference.GetOutputPort())

# Update probe filter to get valid scalar range
probe.Update()
scalar_range = probe.GetOutput().GetScalarRange()

lut = vtk.vtkLookupTable()
lut.SetTableRange(scalar_range)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(probe.GetOutputPort())
mapper.UseLookupTableScalarRangeOn()
mapper.SetLookupTable(lut)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800,600)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()