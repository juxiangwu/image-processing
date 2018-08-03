#coding:utf-8
import vtk
import os
curdir,fname = os.path.split(__file__)

potential_cation = vtk.vtkGaussianCubeReader()
potential_cation.SetFileName(os.path.join(curdir,"/models/crownK.pot.cube"))
potential_cation.SetBScale(20)
potential_cation.SetHBScale(20)
potential_cation.Update()

sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.0)
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)

glyph = vtk.vtkGlyph3D()
glyph.SetInputConnection(potential_cation.GetOutputPort())
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.OrientOn()
glyph.SetColorMode(1)
glyph.SetScaleMode(2)
glyph.ScalingOn()

atomMapper = vtk.vtkPolyDataMapper()
atomMapper.SetInputConnection(glyph.GetOutputPort())

atomActor = vtk.vtkActor()
atomActor.SetMapper(atomMapper)

tube = vtk.vtkTubeFilter()
tube.SetInputConnection(potential_cation.GetOutputPort())
tube.SetNumberOfSides(16)
tube.SetCapping(0)
tube.SetVaryRadius(0)
tube.SetRadius(0.6)
tube.SetRadiusFactor(10)

bondMapper = vtk.vtkPolyDataMapper()
bondMapper.SetInputConnection(tube.GetOutputPort())
bondMapper.SetImmediateModeRendering(1)
bondMapper.UseLookupTableScalarRangeOff()
bondMapper.SetScalarVisibility(1)
bondMapper.SetScalarModeToDefault()

bondActor = vtk.vtkActor()
bondActor.SetMapper(bondMapper)

grad = vtk.vtkImageGradient()
grad.SetDimensionality(3)
grad.SetInputData(potential_cation.GetGridOutput())

attrib = vtk.vtkAssignAttribute()
attrib.SetInputConnection(grad.GetOutputPort())
attrib.Assign(vtk.vtkDataSetAttributes.SCALARS, vtk.vtkDataSetAttributes.VECTORS, vtk.vtkAssignAttribute.POINT_DATA)

center = potential_cation.GetOutput().GetPoint(0)

seeds = vtk.vtkPointSource()
seeds.SetRadius(3.0)
seeds.SetCenter(center)
seeds.SetNumberOfPoints(150)

integ = vtk.vtkRungeKutta4()
stream = vtk.vtkStreamTracer()
stream.SetInputConnection(attrib.GetOutputPort())
stream.SetSourceConnection(seeds.GetOutputPort())
stream.SetMaximumPropagation(500)
#stream.SetStepLength(0.5)
stream.SetInitialIntegrationStep(0.05)
stream.SetIntegrationDirectionToBackward()
stream.SetIntegrator(integ)

streamMapper = vtk.vtkPolyDataMapper()
streamMapper.SetInputConnection(stream.GetOutputPort())

streamActor = vtk.vtkActor()
streamActor.SetMapper(streamMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(atomActor)
renderer.AddActor(bondActor)
renderer.AddActor(streamActor)
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