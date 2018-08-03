#coding:utf-8
import vtk

box = (5.0, 5.0, 5.0)
density = 50
npoints = density**3
spacing = (box[0]/(density-1), box[1]/(density-1), box[2]/(density-1))

def f(x, y, z):
	return (x**2 + y**2) * z**2


grid = vtk.vtkImageData()
grid.SetDimensions(density, density, density)
grid.SetOrigin(-box[0]/2.0, -box[1]/2.0, -box[2]/2.0)
grid.SetSpacing(spacing[0], spacing[1], spacing[2])

data = vtk.vtkFloatArray()
data.SetNumberOfValues(npoints)

for i in range(density):
    z = spacing[0]*i - box[2]/2.0
    for j in range(density):
        y = spacing[1]*j - box[1]/2.0
        for k in range(density):
            x = spacing[2]*k - box[0]/2.0
            n = k + j*density + i*density*density
            data.SetValue(n, f(x, y, z))

grid.GetPointData().SetScalars(data)

planeWidget = vtk.vtkPlaneWidget()
planeWidget.SetInputData(grid)
planeWidget.NormalToZAxisOn()
planeWidget.SetResolution(50)
planeWidget.SetRepresentationToOutline()
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
    global plane, lut, probe
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
window.Render()
interactor.Start()