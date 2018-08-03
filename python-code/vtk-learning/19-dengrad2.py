#coding:utf-8
import vtk
import numpy as N
import os
curdir,fname = os.path.split(__file__)
density = vtk.vtkGaussianCubeReader()
density.SetFileName(os.path.join(curdir,"models/methane-den.cube"))
density.SetBScale(80)
density.SetHBScale(80)
density.Update()

sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.0)
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)

glyph = vtk.vtkGlyph3D()
glyph.SetInputConnection(density.GetOutputPort())
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.OrientOn()
glyph.SetColorMode(1)
glyph.SetScaleMode(2)
glyph.ScalingOn()

atomMapper = vtk.vtkPolyDataMapper()
atomMapper.SetInputConnection(glyph.GetOutputPort())

atomActor = vtk.vtkActor()
atomActor.SetMapper(atomMapper)

grad = vtk.vtkImageGradient()
grad.SetDimensionality(3)
grad.SetInputData(density.GetGridOutput())

attrib = vtk.vtkAssignAttribute()
attrib.SetInputConnection(grad.GetOutputPort())
attrib.Assign(vtk.vtkDataSetAttributes.SCALARS, vtk.vtkDataSetAttributes.VECTORS, vtk.vtkAssignAttribute.POINT_DATA)

C = N.array(density.GetOutput().GetPoint(0))
H = N.array(density.GetOutput().GetPoint(1))

seedsC = vtk.vtkPointSource()
seedsC.SetRadius(0.05)
seedsC.SetCenter(0.9*C+0.1*H)
seedsC.SetNumberOfPoints(30)

seedsH = vtk.vtkPointSource()
seedsH.SetRadius(0.05)
seedsH.SetCenter(0.9*H+0.1*C)
seedsH.SetNumberOfPoints(30)

integ = vtk.vtkRungeKutta4()
streamC = vtk.vtkStreamTracer()
streamC.SetInputConnection(attrib.GetOutputPort())
streamC.SetSourceConnection(seedsC.GetOutputPort())
streamC.SetMaximumPropagation(500)
streamC.SetInitialIntegrationStep(0.05)
streamC.SetIntegrationDirectionToBackward()
streamC.SetIntegrator(integ)

streamCMapper = vtk.vtkPolyDataMapper()
streamCMapper.SetInputConnection(streamC.GetOutputPort())

streamCActor = vtk.vtkActor()
streamCActor.SetMapper(streamCMapper)

streamH = vtk.vtkStreamTracer()
streamH.SetInputConnection(attrib.GetOutputPort())
streamH.SetSourceConnection(seedsH.GetOutputPort())
streamH.SetMaximumPropagation(500)
streamH.SetInitialIntegrationStep(0.05)
streamH.SetIntegrationDirectionToBackward()
streamH.SetIntegrator(integ)

streamHMapper = vtk.vtkPolyDataMapper()
streamHMapper.SetInputConnection(streamH.GetOutputPort())

streamHActor = vtk.vtkActor()
streamHActor.SetMapper(streamHMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(atomActor)
renderer.AddActor(streamCActor)
renderer.AddActor(streamHActor)
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