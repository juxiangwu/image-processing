#coding:utf-8
import vtk
import os
density = vtk.vtkGaussianCubeReader()
curdir,fname = os.path.split(__file__)
model_path = os.path.join(curdir,'models/crown.den.cube')
density.SetFileName(model_path)

density.Update()
grid = density.GetGridOutput()

planeWidget = vtk.vtkPlaneWidget()
planeWidget.SetInputData(grid)
planeWidget.NormalToZAxisOn()
planeWidget.SetResolution(30)
planeWidget.PlaceWidget()

plane = vtk.vtkPolyData()
planeWidget.GetPolyData(plane)

probe = vtk.vtkProbeFilter()
probe.SetInputData(plane)
probe.SetSourceData(grid)

# Update probe filter to get valid scalar range
probe.Update()
scalar_range = probe.GetOutput().GetScalarRange()

lut = vtk.vtkLookupTable()
lut.SetTableRange(scalar_range)
lut.SetScaleToLog10()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(probe.GetOutputPort())
mapper.UseLookupTableScalarRangeOn()
mapper.SetLookupTable(lut)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.VisibilityOff()

outline = vtk.vtkOutlineFilter()
outline.SetInputData(grid)
outlineMapper = vtk.vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())
outlineActor = vtk.vtkActor()
outlineActor.SetMapper(outlineMapper)

renderer = vtk.vtkRenderer()
renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800,600)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

def BeginInteraction(obj, event):
     global plane, actor
     obj.GetPolyData(plane)
     actor.VisibilityOn()

def ProbeData(obj, event):
    global plane, lut
    obj.GetPolyData(plane)
    scalar_range = probe.GetOutput().GetScalarRange()
    lut.SetTableRange(scalar_range)

# Associate the widget with the interactor
planeWidget.SetInteractor(interactor)
# Handle the events.
planeWidget.AddObserver("EnableEvent", BeginInteraction)
planeWidget.AddObserver("StartInteractionEvent", BeginInteraction)
planeWidget.AddObserver("InteractionEvent", ProbeData)

renderer.AddActor(actor)
renderer.AddActor(outlineActor)

interactor.Initialize()
interactor.Start()